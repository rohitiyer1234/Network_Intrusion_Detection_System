`timescale 1ns/1ps

module tb_control_fsm;

    import parameter_pkg::*;

    logic clk;
    logic rst_n;

    logic ctrl_start;
    logic ctrl_reset;

    logic feat_full;
    logic feat_clear;

    logic l1_start;
    logic l1_done;

    logic l2_start;
    logic l2_done;

    logic l3_start;
    logic l3_done;

    logic send_req;
    logic tx_done;

    logic logit_we;

    logic signed [ACC_W-1:0]
        l3_logits [L3_OUT-1:0];

    logic signed [ACC_W-1:0]
        out_logits [L3_OUT-1:0];

    logic status_idle;
    logic status_busy;
    logic status_done;
    logic status_error;

    logic [31:0] inf_counter;
    logic [31:0] latency_cycles;

    // DUT
    control_fsm dut (

        .clk(clk),
        .rst_n(rst_n),

        .ctrl_start(ctrl_start),
        .ctrl_reset(ctrl_reset),

        .feat_full(feat_full),
        .feat_clear(feat_clear),

        .l1_start(l1_start),
        .l1_done(l1_done),

        .l2_start(l2_start),
        .l2_done(l2_done),

        .l3_start(l3_start),
        .l3_done(l3_done),

        .send_req(send_req),
        .tx_done(tx_done),

        .logit_we(logit_we),

        .l3_logits(l3_logits),
        .out_logits(out_logits),

        .status_idle(status_idle),
        .status_busy(status_busy),
        .status_done(status_done),
        .status_error(status_error),

        .inf_counter(inf_counter),
        .latency_cycles(latency_cycles)

    );

    // CLOCK
    initial clk = 0;
    always #5 clk = ~clk;

    integer cycle_ctr;

    always @(posedge clk or negedge rst_n) begin

        if(!rst_n)
            cycle_ctr <= 0;
        else
            cycle_ctr <= cycle_ctr + 1;

    end

    // =========================================================================
    // RESET
    // =========================================================================

    task automatic apply_reset();

        begin

            rst_n = 0;

            ctrl_start = 0;
            ctrl_reset = 0;

            feat_full = 0;

            l1_done = 0;
            l2_done = 0;
            l3_done = 0;

            tx_done = 0;

            for(int i=0;i<L3_OUT;i++)
                l3_logits[i] = i;

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(5) @(posedge clk);

        end

    endtask

    // =========================================================================
    // NORMAL INFERENCE
    // =========================================================================

    task automatic normal_inference();

        integer start_cycle;
        integer end_cycle;

        begin

            $display("TEST : NORMAL INFERENCE");

            feat_full = 1;

            @(posedge clk);

            ctrl_start = 1;

            @(posedge clk);

            ctrl_start = 0;

            wait(l1_start);

            repeat(10) @(posedge clk);

            l1_done = 1;

            @(posedge clk);

            l1_done = 0;

            wait(l2_start);

            repeat(10) @(posedge clk);

            l2_done = 1;

            @(posedge clk);

            l2_done = 0;

            wait(l3_start);

            start_cycle = cycle_ctr;

            repeat(10) @(posedge clk);

            l3_done = 1;

            @(posedge clk);

            l3_done = 0;

            wait(send_req);

            repeat(5) @(posedge clk);

            tx_done = 1;

            @(posedge clk);

            tx_done = 0;

            wait(status_done);

            end_cycle = cycle_ctr;

            $display(
                "TOTAL LATENCY = %0d cycles",
                end_cycle - start_cycle
            );

            $display("NORMAL INFERENCE PASSED");

        end

    endtask

    // =========================================================================
    // TEST : DUPLICATE STARTS
    // =========================================================================

    task automatic duplicate_starts();

        begin

            $display("TEST : DUPLICATE START");

            feat_full = 1;

            @(posedge clk);

            ctrl_start = 1;

            repeat(5) @(posedge clk);

            ctrl_start = 0;

            repeat(20) @(posedge clk);

            $display("DUPLICATE START PASSED");

        end

    endtask

    // =========================================================================
    // TEST : TX BACKPRESSURE
    // =========================================================================

    task automatic tx_backpressure();

        begin

            $display("TEST : TX BACKPRESSURE");

            feat_full = 1;

            @(posedge clk);

            ctrl_start = 1;

            @(posedge clk);

            ctrl_start = 0;

            wait(l1_start);

            @(posedge clk);
            l1_done = 1;
            @(posedge clk);
            l1_done = 0;

            wait(l2_start);

            @(posedge clk);
            l2_done = 1;
            @(posedge clk);
            l2_done = 0;

            wait(l3_start);

            @(posedge clk);
            l3_done = 1;
            @(posedge clk);
            l3_done = 0;

            wait(send_req);

            repeat(30)
                @(posedge clk);

            tx_done = 1;

            @(posedge clk);

            tx_done = 0;

            wait(status_done);

            $display("TX BACKPRESSURE PASSED");

        end

    endtask

    // =========================================================================
    // TEST : TIMEOUT
    // =========================================================================

    task automatic timeout_test();

        begin

            $display("TEST : TIMEOUT");

            feat_full = 1;

            @(posedge clk);

            ctrl_start = 1;

            @(posedge clk);

            ctrl_start = 0;

            repeat(1000010)
                @(posedge clk);

            if(!status_error) begin

                $display("TIMEOUT FAILURE");
                $finish;
            end

            $display("TIMEOUT PASSED");

            ctrl_reset = 1;

            @(posedge clk);

            ctrl_reset = 0;

        end

    endtask

    // =========================================================================
    // TEST : MID COMPUTE RESET
    // =========================================================================

    task automatic mid_reset();

        begin

            $display("TEST : MID RESET");

            feat_full = 1;

            @(posedge clk);

            ctrl_start = 1;

            @(posedge clk);

            ctrl_start = 0;

            repeat(10)
                @(posedge clk);

            ctrl_reset = 1;

            @(posedge clk);

            ctrl_reset = 0;

            repeat(10)
                @(posedge clk);

            if(!status_idle) begin

                $display("MID RESET FAILURE");
                $finish;
            end

            $display("MID RESET PASSED");

        end

    endtask

    // =========================================================================
    // TEST : RANDOM STRESS
    // =========================================================================

    task automatic stress();

        begin

            $display("TEST : STRESS");

            for(int t=0;t<20;t++) begin

                normal_inference();
            end

            $display("STRESS PASSED");

        end

    endtask

    // =========================================================================
    // MAIN
    // =========================================================================

    initial begin

        apply_reset();

        normal_inference();

        duplicate_starts();

        tx_backpressure();

        timeout_test();

        mid_reset();

        stress();

        $display("==================================");
        $display("ALL CONTROL FSM TESTS PASSED");
        $display("==================================");

        $finish;

    end

endmodule