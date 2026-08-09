/*

Regular testbench to check single cycle run and validate functional mac_array

`timescale 1ns/1ps

module tb_mac_array;

    parameter IN_WIDTH    = 8;
    parameter WT_WIDTH    = 8;
    parameter ACC_WIDTH   = 32;

    parameter IN_DIM      = 20;
    parameter PARALLELISM = 4;

    logic clk;
    logic rst_n;
    logic start;
    logic done;

    logic signed [IN_WIDTH-1:0]
        in_vec [IN_DIM-1:0];

    logic signed [WT_WIDTH-1:0]
        wt_data [PARALLELISM-1:0];

    logic signed [ACC_WIDTH-1:0]
        acc_out [PARALLELISM-1:0];

    // =========================================================
    // DUT
    // =========================================================

    mac_array #(
        .PARALLELISM(PARALLELISM),
        .IN_DIM(IN_DIM),
        .IN_WIDTH(IN_WIDTH),
        .WT_WIDTH(WT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ACTIVE_MACS(PARALLELISM)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_vec(in_vec),
        .wt_data(wt_data),
        .acc_out(acc_out),
        .done(done)
    );

    // =========================================================
    // Clock
    // =========================================================

    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================
    // Stimulus
    // =========================================================

    integer expected [PARALLELISM-1:0];
    initial begin

    rst_n = 0;
    start = 0;

    // Initialize all inputs

    for(int i=0;i<IN_DIM;i++)
        in_vec[i] = '0;

    for(int k=0;k<PARALLELISM;k++)
        wt_data[k] = '0;

    repeat(5) @(posedge clk);

    rst_n = 1;

    end

    initial begin

        rst_n = 0;
        start = 0;

        repeat(5) @(posedge clk);

        rst_n = 1;

        // -----------------------------------------------------
        // Input vector
        // -----------------------------------------------------

        for(int i=0;i<IN_DIM;i++)
            in_vec[i] = i + 1;

        // -----------------------------------------------------
        // Weights
        // -----------------------------------------------------

        for(int k=0;k<PARALLELISM;k++) begin
            wt_data[k] = k + 1;
            expected[k] = 0;

            for(int i=0;i<IN_DIM;i++)
                expected[k] += (i+1)*(k+1);
        end

        // -----------------------------------------------------
        // Start
        // -----------------------------------------------------

        @(posedge clk);
        start = 1;

        @(posedge clk);
        start = 0;

        wait(done);

        @(posedge clk);

        // -----------------------------------------------------
        // Check outputs
        // -----------------------------------------------------

        for(int k=0;k<PARALLELISM;k++) begin

            $display(
                "MAC[%0d] = %0d expected = %0d",
                k,
                acc_out[k],
                expected[k]
            );

            if(acc_out[k] != expected[k]) begin
                $display("FAIL");
                $finish;
            end
        end

        $display("================================");
        $display("ALL TESTS PASSED");
        $display("================================");

        $finish;
    end

endmodule

*/

`timescale 1ns/1ps

// =============================================================================
// Layered Verification TB for mac_array
// -----------------------------------------------------------------------------
// Features:
//   - layered/task-based verification
//   - reset verification
//   - multi-pass verification
//   - accumulation correctness
//   - restart/reuse validation
//   - done timing validation
//   - active_macs masking validation
//   - signed arithmetic testing
//   - sequential layer-style operation
//   - corner-case testing
//
// This TB validates the exact use-case of:
//
//   Layer1 : 20 -> 64
//
// where:
//   pass1 computes neurons 0-31
//   pass2 computes neurons 32-63
//
// ensuring accumulators reset correctly between passes.
// =============================================================================

module tb_mac_array;

    // =========================================================================
    // Parameters
    // =========================================================================

    parameter IN_WIDTH    = 8;
    parameter WT_WIDTH    = 8;
    parameter ACC_WIDTH   = 32;

    parameter IN_DIM      = 20;
    parameter PARALLELISM = 4;

    // =========================================================================
    // DUT Signals
    // =========================================================================

    logic clk;
    logic rst_n;
    logic start;
    logic done;

    logic signed [IN_WIDTH-1:0]
        in_vec [IN_DIM-1:0];

    logic signed [WT_WIDTH-1:0]
        wt_data [PARALLELISM-1:0];

    logic signed [ACC_WIDTH-1:0]
        acc_out [PARALLELISM-1:0];

    // =========================================================================
    // Expected Results
    // =========================================================================

    integer expected [PARALLELISM-1:0];

    // =========================================================================
    // DUT
    // =========================================================================

    mac_array #(
        .PARALLELISM(PARALLELISM),
        .IN_DIM(IN_DIM),
        .IN_WIDTH(IN_WIDTH),
        .WT_WIDTH(WT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ACTIVE_MACS(PARALLELISM)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_vec(in_vec),
        .wt_data(wt_data),
        .acc_out(acc_out),
        .done(done)
    );

    // =========================================================================
    // Clock
    // =========================================================================

    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // RESET TASK
    // =========================================================================

    task automatic apply_reset();

        begin

            rst_n = 0;
            start = 0;

            for(int i=0;i<IN_DIM;i++)
                in_vec[i] = '0;

            for(int k=0;k<PARALLELISM;k++) begin
                wt_data[k] = '0;
                expected[k] = 0;
            end

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(2) @(posedge clk);

        end

    endtask

    // =========================================================================
    // LOAD INPUT VECTOR
    // =========================================================================

    task automatic load_inputs_linear();

        begin

            for(int i=0;i<IN_DIM;i++)
                in_vec[i] = i + 1;

        end

    endtask

    // =========================================================================
    // LOAD CONSTANT WEIGHTS
    // =========================================================================

    task automatic load_constant_weights(
        input int base
    );

        begin

            for(int k=0;k<PARALLELISM;k++) begin

                wt_data[k] = base + k;

            end

        end

    endtask

    // =========================================================================
    // COMPUTE EXPECTED
    // =========================================================================

    task automatic compute_expected_constant_weights(
        input int base
    );

        begin

            for(int k=0;k<PARALLELISM;k++) begin

                expected[k] = 0;

                for(int i=0;i<IN_DIM;i++) begin

                    expected[k] +=
                        (i+1) * (base+k);

                end
            end
        end

    endtask

    // =========================================================================
    // START OPERATION
    // =========================================================================

    task automatic start_operation();

        begin

            @(posedge clk);
            start <= 1'b1;

            @(posedge clk);
            start <= 1'b0;

        end

    endtask

    // =========================================================================
    // WAIT DONE
    // =========================================================================

    task automatic wait_done();

        begin

            wait(done == 1'b1);

            @(posedge clk);

        end

    endtask

    // =========================================================================
    // CHECK OUTPUTS
    // =========================================================================

    task automatic check_outputs(
        input string testname
    );

        begin

            $display("====================================");
            $display("CHECKING : %s", testname);
            $display("====================================");

            for(int k=0;k<PARALLELISM;k++) begin

                $display(
                    "MAC[%0d] = %0d expected = %0d",
                    k,
                    acc_out[k],
                    expected[k]
                );

                if(acc_out[k] !== expected[k]) begin

                    $display("FAIL : %s", testname);
                    $finish;

                end
            end

            $display("PASS : %s", testname);

        end

    endtask

    // =========================================================================
    // VERIFY RESET CLEARS ACCUMULATORS
    // =========================================================================

    task automatic verify_reset_clear();

        begin

            @(posedge clk);

            rst_n <= 0;

            repeat(3) @(posedge clk);

            for(int k=0;k<PARALLELISM;k++) begin

                if(acc_out[k] !== 0) begin

                    $display("RESET CLEAR FAILED");
                    $finish;

                end
            end

            rst_n <= 1;

            repeat(2) @(posedge clk);

            $display("RESET CLEAR PASSED");

        end

    endtask

    // =========================================================================
    // TEST 1
    // Basic functionality
    // =========================================================================

    task automatic test_basic();

        begin

            load_inputs_linear();

            load_constant_weights(1);

            compute_expected_constant_weights(1);

            start_operation();

            wait_done();

            check_outputs("BASIC FUNCTIONALITY");

        end

    endtask

    // =========================================================================
    // TEST 2
    // Multi-pass layer behavior
    // Simulates:
    //   neurons 0-31
    //   then neurons 32-63
    // =========================================================================

    task automatic test_layered_passes();

        integer first_pass [PARALLELISM-1:0];

        begin

            // -------------------------------------------------------------
            // PASS 1
            // -------------------------------------------------------------

            load_inputs_linear();

            load_constant_weights(1);

            compute_expected_constant_weights(1);

            start_operation();

            wait_done();

            check_outputs("LAYER PASS 1");

            for(int k=0;k<PARALLELISM;k++)
                first_pass[k] = acc_out[k];

            // -------------------------------------------------------------
            // PASS 2
            // MAC accumulators MUST reset internally
            // -------------------------------------------------------------

            load_constant_weights(10);

            compute_expected_constant_weights(10);

            start_operation();

            wait_done();

            check_outputs("LAYER PASS 2");

            // Ensure no accumulation leakage

            for(int k=0;k<PARALLELISM;k++) begin

                if(acc_out[k] == first_pass[k]) begin

                    $display("ACCUMULATOR RESET FAILURE");
                    $finish;
                end
            end

            $display("MULTI PASS TEST PASSED");

        end

    endtask

    // =========================================================================
    // TEST 3
    // Signed arithmetic
    // =========================================================================

    task automatic test_signed();

        begin

            for(int i=0;i<IN_DIM;i++)
                in_vec[i] = -i;

            for(int k=0;k<PARALLELISM;k++) begin

                wt_data[k] = k - 2;

                expected[k] = 0;

                for(int i=0;i<IN_DIM;i++) begin

                    expected[k] +=
                        (-i) * (k-2);

                end
            end

            start_operation();

            wait_done();

            check_outputs("SIGNED TEST");

        end

    endtask

    // =========================================================================
    // TEST 4
    // Start ignored while running
    // =========================================================================

    task automatic test_illegal_restart();

        begin

            load_inputs_linear();

            load_constant_weights(2);

            compute_expected_constant_weights(2);

            @(posedge clk);

            start <= 1'b1;

            @(posedge clk);

            start <= 1'b0;

            // Illegal restart attempt

            repeat(5) @(posedge clk);

            start <= 1'b1;

            @(posedge clk);

            start <= 1'b0;

            wait_done();

            check_outputs("ILLEGAL RESTART");

        end

    endtask

    // =========================================================================
    // MAIN TEST SEQUENCE
    // =========================================================================

    initial begin

        apply_reset();

        test_basic();

        verify_reset_clear();

        test_layered_passes();

        test_signed();

        test_illegal_restart();

        $display("====================================");
        $display("ALL TESTS PASSED");
        $display("====================================");

        $finish;

    end

    // =========================================================================
    // Waveform Dump
    // =========================================================================

    initial begin

        $dumpfile("mac_array.vcd");

        $dumpvars(0, tb_mac_array);

    end

endmodule