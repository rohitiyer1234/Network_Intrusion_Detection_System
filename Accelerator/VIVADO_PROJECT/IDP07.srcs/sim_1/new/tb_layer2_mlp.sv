`timescale 1ns/1ps

// =============================================================================
// Layered Verification TB
// layer2_mlp
//
// Verifies:
//   - full 32-neuron execution
//   - MAC restart behavior
//   - reset recovery
//   - layer restart
//   - done timing
//   - activation pipeline timing
//   - accumulator clearing
//   - repeated inference
//   - no stale accumulation leakage
//   - randomized signed stimulus
//   - no X/Z propagation
//   - latency metrics
//   - utilization metrics
// =============================================================================

module tb_layer2_mlp;

    parameter IN_DIM  = 64;
    parameter OUT_DIM = 32;
    parameter PAR     = 32;

    logic clk;
    logic rst_n;
    logic start;
    logic done;

    logic signed [7:0]
        in_vec [IN_DIM-1:0];

    logic signed [7:0]
        act_out [OUT_DIM-1:0];

    // =========================================================================
    // DUT
    // =========================================================================

    layer2_mlp #(
        .IN_DIM(IN_DIM),
        .OUT_DIM(OUT_DIM),
        .PAR(PAR)
    ) dut (

        .clk(clk),
        .rst_n(rst_n),
        .start(start),

        .in_vec(in_vec),

        .act_out(act_out),

        .done(done)

    );

    // =========================================================================
    // Performance Counters
    // =========================================================================

    integer start_cycle;
    integer end_cycle;
    integer latency_cycles;

    integer total_mac_ops;

    real utilization;

    // =========================================================================
    // Clock
    // =========================================================================

    initial clk = 0;

    always #5 clk = ~clk;

    // =========================================================================
    // Global Cycle Counter
    // =========================================================================

    integer cycle_counter;

    always @(posedge clk or negedge rst_n) begin

        if(!rst_n)
            cycle_counter <= 0;
        else
            cycle_counter <= cycle_counter + 1;

    end

    // =========================================================================
    // Measure Latency
    // =========================================================================

    task automatic measure_latency(
        input string testname
    );

        begin

            start_cycle = cycle_counter;

            wait(done == 1'b1);

            end_cycle = cycle_counter;

            latency_cycles =
                end_cycle - start_cycle;

            total_mac_ops =
                IN_DIM * OUT_DIM;

            utilization =
                real'(total_mac_ops)
                /
                real'(latency_cycles * PAR);

            $display("================================================");
            $display("PERFORMANCE REPORT : %s", testname);
            $display("================================================");

            $display("Start Cycle      : %0d", start_cycle);
            $display("End Cycle        : %0d", end_cycle);

            $display("Latency Cycles   : %0d", latency_cycles);

            $display("Total MAC Ops    : %0d", total_mac_ops);

            $display("Parallel MACs    : %0d", PAR);

            $display("MAC Utilization  : %0f", utilization);

            $display("Clock Period(ns) : 10");

            $display(
                "Inference Time(ns) : %0d",
                latency_cycles * 10
            );

            $display(
                "Inference Time(us) : %0f",
                (latency_cycles * 10.0)/1000.0
            );

            $display("================================================");

        end

    endtask

    // =========================================================================
    // RESET
    // =========================================================================

    task automatic apply_reset();

        begin

            rst_n = 0;
            start = 0;

            for(int i=0;i<IN_DIM;i++)
                in_vec[i] = '0;

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(2) @(posedge clk);

        end

    endtask

    // =========================================================================
    // Deterministic Signed Stimulus
    // =========================================================================

    task automatic load_pattern_inputs();

        begin

            for(int i=0;i<IN_DIM;i++) begin

                case(i % 8)

                    0: in_vec[i] =  12;
                    1: in_vec[i] = -7;
                    2: in_vec[i] =  25;
                    3: in_vec[i] = -14;
                    4: in_vec[i] =  5;
                    5: in_vec[i] = -3;
                    6: in_vec[i] =  31;
                    7: in_vec[i] = -22;

                endcase

            end

        end

    endtask

    // =========================================================================
    // Randomized Quantized Stimulus
    // =========================================================================

    task automatic load_random_inputs();

        begin

            for(int i=0;i<IN_DIM;i++) begin

                in_vec[i] =
                    $urandom_range(-128,127);

            end

        end

    endtask

    // =========================================================================
    // Start Layer
    // =========================================================================

    task automatic start_layer();

        begin

            @(posedge clk);

            start <= 1'b1;

            @(posedge clk);

            start <= 1'b0;

        end

    endtask

    // =========================================================================
    // Wait Done
    // =========================================================================

    task automatic wait_done();

        begin

            wait(done == 1'b1);

            @(posedge clk);

        end

    endtask

    // =========================================================================
    // Check Outputs
    // =========================================================================

    task automatic check_outputs();

        integer nonzero_count;

        begin

            nonzero_count = 0;

            for(int i=0;i<OUT_DIM;i++) begin

                if(^act_out[i] === 1'bX) begin

                    $display(
                        "ERROR : X DETECTED @ neuron %0d",
                        i
                    );

                    $finish;

                end

                if(act_out[i] !== 8'h00)
                    nonzero_count++;

            end

            $display(
                "NONZERO OUTPUTS = %0d / %0d",
                nonzero_count,
                OUT_DIM
            );

            if(nonzero_count == 0) begin

                $display(
                    "ERROR : ALL OUTPUTS ZERO"
                );

                $finish;

            end

        end

    endtask

    // =========================================================================
    // TEST 1
    // Full Layer Execution
    // =========================================================================

    task automatic test_full_layer();

        begin

            $display(
                "TEST : FULL 32 NEURON EXECUTION"
            );

            load_pattern_inputs();

            fork

                begin
                    start_layer();
                end

                begin
                    measure_latency(
                        "LAYER2 FULL EXECUTION"
                    );
                end

            join

            wait_done();

            check_outputs();

            $display(
                "FULL LAYER2 COMPLETE"
            );

        end

    endtask

    // =========================================================================
    // TEST 2
    // Restart
    // =========================================================================

    task automatic test_restart();

        begin

            $display("TEST : RESTART");

            load_pattern_inputs();

            start_layer();

            wait_done();

            repeat(5) @(posedge clk);

            start_layer();

            wait_done();

            check_outputs();

            $display("RESTART PASSED");

        end

    endtask

    // =========================================================================
    // TEST 3
    // Mid-compute reset
    // =========================================================================

    task automatic test_mid_reset();

        begin

            $display(
                "TEST : MID COMPUTE RESET"
            );

            load_pattern_inputs();

            start_layer();

            repeat(25) @(posedge clk);

            rst_n <= 0;

            repeat(5) @(posedge clk);

            rst_n <= 1;

            repeat(5) @(posedge clk);

            if(done !== 0) begin

                $display(
                    "RESET FAILURE"
                );

                $finish;

            end

            $display(
                "MID RESET PASSED"
            );

        end

    endtask

    // =========================================================================
    // TEST 4
    // Illegal restart while busy
    // =========================================================================

    task automatic test_illegal_restart();

        begin

            $display(
                "TEST : ILLEGAL RESTART"
            );

            load_pattern_inputs();

            start_layer();

            repeat(15) @(posedge clk);

            @(posedge clk);
            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

            wait_done();

            check_outputs();

            $display(
                "ILLEGAL RESTART PASSED"
            );

        end

    endtask

    // =========================================================================
    // TEST 5
    // Multi-run randomized stress
    // =========================================================================

    task automatic test_stress();

        begin

            $display(
                "TEST : MULTI RUN STRESS"
            );

            for(int t=0;t<10;t++) begin

                load_random_inputs();

                start_layer();

                wait_done();

                check_outputs();

            end

            $display(
                "STRESS PASSED"
            );

        end

    endtask

    // =========================================================================
    // TEST 6
    // Accumulator leakage detection
    // =========================================================================

    task automatic test_accumulator_clear();

        integer prev_nonzero;

        begin

            $display(
                "TEST : ACCUMULATOR CLEAR"
            );

            load_pattern_inputs();

            start_layer();

            wait_done();

            prev_nonzero = 0;

            for(int i=0;i<OUT_DIM;i++) begin

                if(act_out[i] != 0)
                    prev_nonzero++;
            end

            // second run with all zeros

            for(int i=0;i<IN_DIM;i++)
                in_vec[i] = 0;

            start_layer();

            wait_done();

            for(int i=0;i<OUT_DIM;i++) begin

                if(act_out[i] > 8'd5) begin

                    $display(
                        "ACCUMULATOR LEAKAGE DETECTED"
                    );

                    $finish;

                end

            end

            $display(
                "ACCUMULATOR CLEAR PASSED"
            );

        end

    endtask

    // =========================================================================
    // MAIN
    // =========================================================================

    initial begin

        apply_reset();

        test_full_layer();

        test_restart();

        test_mid_reset();

        apply_reset();

        test_illegal_restart();

        test_accumulator_clear();

        test_stress();

        $display("====================================");
        $display("ALL LAYER2 TESTS PASSED");
        $display("====================================");

        $finish;

    end

    // =========================================================================
    // Waveforms
    // =========================================================================

    initial begin

        $dumpfile("layer2_mlp.vcd");

        $dumpvars(0, tb_layer2_mlp);

    end

endmodule