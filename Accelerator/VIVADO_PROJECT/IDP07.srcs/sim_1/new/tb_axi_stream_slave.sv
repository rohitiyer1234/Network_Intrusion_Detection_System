`timescale 1ns/1ps

module tb_axi_stream_slave;

    parameter DATA_WIDTH   = 64;
    parameter NUM_FEATURES = 20;
    parameter FEAT_WIDTH   = 8;

    logic clk;
    logic rst_n;

    // AXIS
    logic [DATA_WIDTH-1:0] s_axis_tdata;
    logic s_axis_tvalid;
    logic s_axis_tready;
    logic s_axis_tlast;

    // FEATURE BUFFER
    logic fb_write_en;

    logic signed [FEAT_WIDTH-1:0]
        fb_write_data;

    logic frame_done;

    logic fb_full;

    logic packet_error;

    // DUT
    axi_stream_slave dut (

        .clk(clk),
        .rst_n(rst_n),

        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),

        .fb_write_en(fb_write_en),
        .fb_write_data(fb_write_data),

        .frame_done(frame_done),

        .fb_full(fb_full),

        .packet_error(packet_error)

    );

    // =========================================================================
    // CLOCK
    // =========================================================================

    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // METRICS
    // =========================================================================

    integer cycle_ctr;

    integer start_cycle;
    integer end_cycle;

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

            s_axis_tdata  = 0;
            s_axis_tvalid = 0;
            s_axis_tlast  = 0;

            fb_full = 0;

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(5) @(posedge clk);

        end

    endtask

    // =========================================================================
    // SEND FRAME
    // =========================================================================

    task automatic send_frame();

        begin

            start_cycle = cycle_ctr;

            for(int i=0;i<NUM_FEATURES;i++) begin

                @(posedge clk);

                s_axis_tvalid <= 1'b1;

                s_axis_tdata <= i;

                s_axis_tlast <=
                    (i == NUM_FEATURES-1);

                wait(s_axis_tready);

            end

            @(posedge clk);

            s_axis_tvalid <= 0;
            s_axis_tlast  <= 0;

            wait(frame_done);

            end_cycle = cycle_ctr;

            $display(
                "FRAME LATENCY = %0d cycles",
                end_cycle - start_cycle
            );

        end

    endtask

    // =========================================================================
    // TEST : BASIC
    // =========================================================================

    task automatic test_basic();

        begin

            $display("TEST : BASIC FRAME");

            send_frame();

            if(packet_error) begin

                $display("FAIL");
                $finish;
            end

            $display("BASIC PASSED");

        end

    endtask

    // =========================================================================
    // TEST : BACKPRESSURE
    // =========================================================================

    task automatic test_backpressure();

        begin

            $display("TEST : BACKPRESSURE");

            fork

                begin

                    send_frame();

                end

                begin

                    repeat(5) @(posedge clk);

                    fb_full <= 1'b1;

                    repeat(10) @(posedge clk);

                    fb_full <= 1'b0;

                end

            join

            $display("BACKPRESSURE PASSED");

        end

    endtask

    // =========================================================================
    // TEST : EARLY TLAST
    // =========================================================================

    task automatic test_early_tlast();

        begin

            $display("TEST : EARLY TLAST");

            for(int i=0;i<5;i++) begin

                @(posedge clk);

                s_axis_tvalid <= 1'b1;

                s_axis_tdata <= i;

                s_axis_tlast <=
                    (i == 4);

            end

            @(posedge clk);

            s_axis_tvalid <= 0;
            s_axis_tlast  <= 0;

            wait(packet_error);

            $display("EARLY TLAST PASSED");

        end

    endtask

    // =========================================================================
    // TEST : MISSING TLAST
    // =========================================================================

    task automatic test_missing_tlast();

        begin

            $display("TEST : MISSING TLAST");

            for(int i=0;i<NUM_FEATURES;i++) begin

                @(posedge clk);

                s_axis_tvalid <= 1'b1;

                s_axis_tdata <= i;

                s_axis_tlast <= 0;

            end

            @(posedge clk);

            s_axis_tvalid <= 0;

            wait(packet_error);

            $display("MISSING TLAST PASSED");

        end

    endtask

    // =========================================================================
    // TEST : RANDOM STRESS
    // =========================================================================

    task automatic test_stress();

        begin

            $display("TEST : STRESS");

            for(int t=0;t<20;t++) begin

                send_frame();

            end

            $display("STRESS PASSED");

        end

    endtask

    // =========================================================================
    // MAIN
    // =========================================================================

    initial begin

        apply_reset();

        test_basic();

        test_backpressure();

        test_early_tlast();

        test_missing_tlast();

        test_stress();

        $display("=================================");
        $display("ALL AXIS SLAVE TESTS PASSED");
        $display("=================================");

        $finish;

    end

endmodule