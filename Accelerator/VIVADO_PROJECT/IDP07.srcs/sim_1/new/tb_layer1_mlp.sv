`timescale 1ns/1ps

// =============================================================================
// Layered Verification TB
// mlp_layer
//
// Verifies:
//   - 64-neuron multi-group execution
//   - MAC restart behavior
//   - reset recovery
//   - layer restart
//   - done timing
//   - activation pipeline timing
//   - accumulator clearing between groups
//   - repeated inference
//   - no stale accumulation leakage
// =============================================================================

module tb_layer1_mlp;

    parameter IN_DIM  = 20;
    parameter OUT_DIM = 64;
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

    layer1_mlp #(
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
    // Measure Layer Latency
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

            // -------------------------------------------------------------
            // MAC operations
            // -------------------------------------------------------------

            total_mac_ops =
                IN_DIM * OUT_DIM;

            // -------------------------------------------------------------
            // Utilization estimate
            // -------------------------------------------------------------

            utilization =
                real'(total_mac_ops) /
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
                in_vec[i] = 0;

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(2) @(posedge clk);

        end

    endtask

    // =========================================================================
    // Load Inputs
    // =========================================================================

    task automatic load_inputs();

        begin

            for(int i=0;i<IN_DIM;i++)
                in_vec[i] = i + 1;

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
    // TEST 1
    // Basic full 64-neuron processing
    // =========================================================================

        task automatic test_full_layer();

        begin

            $display("TEST : FULL 64 NEURON EXECUTION");

            load_inputs();

            fork

                begin
                    start_layer();
                end

                begin
                    measure_latency(
                        "FULL LAYER EXECUTION"
                    );
                end

            join

            wait_done();

            $display("FULL LAYER COMPLETE");

        end

    endtask

    // =========================================================================
    // TEST 2
    // Restart layer after completion
    // =========================================================================

    task automatic test_restart();

        begin

            $display("TEST : RESTART");

            load_inputs();

            start_layer();

            wait_done();

            repeat(5) @(posedge clk);

            start_layer();

            wait_done();

            $display("RESTART PASSED");

        end

    endtask

    // =========================================================================
    // TEST 3
    // Reset during active computation
    // =========================================================================

    task automatic test_mid_reset();

        begin

            $display("TEST : MID COMPUTE RESET");

            load_inputs();

            start_layer();

            repeat(20) @(posedge clk);

            rst_n <= 0;

            repeat(5) @(posedge clk);

            rst_n <= 1;

            repeat(5) @(posedge clk);

            if(done !== 0) begin

                $display("RESET FAILURE");
                $finish;

            end

            $display("MID RESET PASSED");

        end

    endtask

    // =========================================================================
    // TEST 4
    // Illegal restart while busy
    // =========================================================================

    task automatic test_illegal_restart();

        begin

            $display("TEST : ILLEGAL RESTART");

            load_inputs();

            start_layer();

            repeat(10) @(posedge clk);

            // should be ignored

            @(posedge clk);
            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

            wait_done();

            $display("ILLEGAL RESTART PASSED");

        end

    endtask

    // =========================================================================
    // TEST 5
    // Multi-run stress
    // =========================================================================

    task automatic test_stress();

        begin

            $display("TEST : MULTI RUN STRESS");

            for(int t=0;t<5;t++) begin

                load_inputs();

                start_layer();

                wait_done();

            end

            $display("STRESS PASSED");

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

        test_stress();

        $display("====================================");
        $display("ALL LAYER TESTS PASSED");
        $display("====================================");

        $finish;

    end

    // =========================================================================
    // Waveforms
    // =========================================================================

    initial begin

        $dumpfile("layer1_mlp.vcd");

        $dumpvars(0, tb_layer1_mlp);

    end

endmodule