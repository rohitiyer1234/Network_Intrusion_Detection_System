`timescale 1ns/1ps

module tb_mac_unit;

    parameter IN_WIDTH  = 8;
    parameter WT_WIDTH  = 8;
    parameter ACC_WIDTH = 32;

    logic clk;
    logic rst_n;
    logic en;
    logic clear_acc;

    logic signed [IN_WIDTH-1:0] a;
    logic signed [WT_WIDTH-1:0] w;

    logic signed [ACC_WIDTH-1:0] acc;

    // =========================================================
    // DUT
    // =========================================================

    mac_unit #(
        .IN_WIDTH(IN_WIDTH),
        .WT_WIDTH(WT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .clear_acc(clear_acc),
        .a(a),
        .w(w),
        .acc(acc)
    );

    // =========================================================
    // Clock Generation
    // =========================================================

    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================
    // Test Sequence
    // =========================================================

    initial begin

        $display("=======================================");
        $display("Starting MAC Unit Test");
        $display("=======================================");

        rst_n = 0;
        en = 0;
        clear_acc = 0;
        a = 0;
        w = 0;

        repeat(5) @(posedge clk);

        rst_n = 1;

        // -----------------------------------------------------
        // TEST 1 : First Multiply
        // -----------------------------------------------------

        @(posedge clk);
        en = 1;
        clear_acc = 1;
        a = 10;
        w = 5;

        @(posedge clk);
        clear_acc = 0;
        a = 3;
        w = 4;

        @(posedge clk);
        a = -2;
        w = 8;

        @(posedge clk);
        a = 7;
        w = -3;

        @(posedge clk);
        en = 0;
        a = 0;
        w = 0;

        repeat(5) @(posedge clk);

        $display("Final ACC = %0d", acc);

        // Expected:
        //
        // 10*5   = 50
        // 3*4    = 12
        // -2*8   = -16
        // 7*-3   = -21
        //
        // Total = 25

        if (acc == 25)
            $display("TEST PASSED");
        else
            $display("TEST FAILED");

        // -----------------------------------------------------
        // TEST 2 : enable gap handling
        // -----------------------------------------------------

        @(posedge clk);
        en = 1;
        clear_acc = 1;
        a = 2;
        w = 2;

        @(posedge clk);
        en = 0;

        repeat(3) @(posedge clk);

        @(posedge clk);
        en = 1;
        clear_acc = 0;
        a = 5;
        w = 5;

        @(posedge clk);
        en = 0;

        repeat(4) @(posedge clk);

        $display("ACC after enable-gap test = %0d", acc);

        // Expected:
        // 2*2 + 5*5 = 29

        if (acc == 29)
            $display("ENABLE GAP TEST PASSED");
        else
            $display("ENABLE GAP TEST FAILED");

        // -----------------------------------------------------
        // TEST 3 : Signed arithmetic
        // -----------------------------------------------------

        @(posedge clk);
        en = 1;
        clear_acc = 1;
        a = -10;
        w = -10;

        @(posedge clk);
        clear_acc = 0;
        a = -5;
        w = 3;

        @(posedge clk);
        en = 0;

        repeat(4) @(posedge clk);

        $display("Signed arithmetic ACC = %0d", acc);

        // Expected:
        // (-10 * -10) + (-5 * 3)
        // 100 - 15 = 85

        if (acc == 85)
            $display("SIGNED TEST PASSED");
        else
            $display("SIGNED TEST FAILED");

        $display("=======================================");
        $display("Simulation Finished");
        $display("=======================================");

        $finish;
    end

    // =========================================================
    // Waveform Dump
    // =========================================================

    initial begin
        $dumpfile("mac_unit.vcd");
        $dumpvars(0, tb_mac_unit);
    end

endmodule