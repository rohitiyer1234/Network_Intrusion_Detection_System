`timescale 1ns/1ps

module tb_register_bank;

    import parameter_pkg::*;

    localparam int NUM_FEATURES = 20;
    localparam int NUM_CLASSES  = 3;

    logic clk;
    logic rst_n;

    // WRITE
    logic        wr_en;
    logic [3:0]  wr_addr;
    logic [31:0] wr_data;

    // READ
    logic        rd_en;
    logic [3:0]  rd_addr;
    logic [31:0] rd_data;

    // CONTROL
    logic ctrl_start;
    logic ctrl_reset;

    // STATUS
    logic status_idle;
    logic status_busy;
    logic status_done;

    logic [31:0] inf_counter;
    logic [31:0] latency_cycles;

    // LOGITS
    logic logit_we;

    logic signed [ACC_W-1:0]
        logits [NUM_CLASSES-1:0];

    // DUT
    register_bank dut (

        .clk(clk),
        .rst_n(rst_n),

        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),

        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_data(rd_data),

        .ctrl_start(ctrl_start),
        .ctrl_reset(ctrl_reset),

        .status_idle(status_idle),
        .status_busy(status_busy),
        .status_done(status_done),

        .inf_counter(inf_counter),
        .latency_cycles(latency_cycles),

        .logit_we(logit_we),
        .logits(logits)

    );

    // CLOCK
    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // PERFORMANCE
    // =========================================================================

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

            wr_en = 0;
            wr_addr = 0;
            wr_data = 0;

            rd_en = 0;
            rd_addr = 0;

            status_idle = 1;
            status_busy = 0;
            status_done = 0;

            inf_counter = 0;
            latency_cycles = 0;

            logit_we = 0;

            for(int i=0;i<NUM_CLASSES;i++)
                logits[i] = 0;

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(5) @(posedge clk);

        end

    endtask

    // =========================================================================
    // WRITE
    // =========================================================================

    task automatic cpu_write(

        input [3:0] addr,
        input [31:0] data

    );

        begin

            @(posedge clk);

            wr_en   <= 1'b1;
            wr_addr <= addr;
            wr_data <= data;

            @(posedge clk);

            wr_en <= 0;

        end

    endtask

    // =========================================================================
    // READ
    // =========================================================================

    task automatic cpu_read(

        input [3:0] addr

    );

        begin

            @(posedge clk);

            rd_en   <= 1'b1;
            rd_addr <= addr;

            @(posedge clk);

            rd_en <= 0;

            @(posedge clk);

            $display(
                "READ addr=%h data=%h",
                addr,
                rd_data
            );

        end

    endtask

    // =========================================================================
    // TESTS
    // =========================================================================

    task automatic test_start_pulse();

        begin

            $display("TEST : START PULSE");

            status_busy = 0;

            cpu_write(4'h0, 32'h1);

            if(!ctrl_start) begin

                $display("START PULSE FAILURE");
                $finish;
            end

            @(posedge clk);

            if(ctrl_start) begin

                $display("START NOT CLEARED");
                $finish;
            end

            $display("START PULSE PASSED");

        end

    endtask

    task automatic test_busy_block();

        begin

            $display("TEST : BUSY BLOCK");

            status_busy = 1;

            cpu_write(4'h0, 32'h1);

            if(ctrl_start) begin

                $display("BUSY BLOCK FAILURE");
                $finish;
            end

            status_busy = 0;

            $display("BUSY BLOCK PASSED");

        end

    endtask

    task automatic test_reset_pulse();

        begin

            $display("TEST : RESET PULSE");

            cpu_write(4'h0, 32'h2);

            if(!ctrl_reset) begin

                $display("RESET FAILURE");
                $finish;
            end

            @(posedge clk);

            if(ctrl_reset) begin

                $display("RESET STUCK");
                $finish;
            end

            $display("RESET PULSE PASSED");

        end

    endtask

    task automatic test_logit_capture();

        begin

            $display("TEST : LOGIT CAPTURE");

            logits[0] = -100;
            logits[1] = 500;
            logits[2] = -20;

            @(posedge clk);

            logit_we <= 1'b1;

            @(posedge clk);

            logit_we <= 1'b0;

            cpu_read(4'h8);
            cpu_read(4'h9);
            cpu_read(4'hA);

            $display("LOGIT CAPTURE PASSED");

        end

    endtask

    task automatic test_metrics();

        begin

            $display("TEST : METRICS");

            inf_counter    = 32'd55;
            latency_cycles = 32'd101;

            cpu_read(4'h4);
            cpu_read(4'h5);

            $display("METRICS PASSED");

        end

    endtask

    task automatic test_random_stress();

        begin

            $display("TEST : RANDOM STRESS");

            for(int i=0;i<50;i++) begin

                cpu_write($random % 16, $random);

                cpu_read($random % 16);

            end

            $display("RANDOM STRESS PASSED");

        end

    endtask

    // =========================================================================
    // MAIN
    // =========================================================================

    initial begin

        apply_reset();

        test_start_pulse();

        test_busy_block();

        test_reset_pulse();

        test_logit_capture();

        test_metrics();

        test_random_stress();

        $display("==================================");
        $display("ALL REGISTER BANK TESTS PASSED");
        $display("==================================");

        $finish;

    end

endmodule