`timescale 1ns/1ps

module tb_axi_lite_slave;

    logic clk;
    logic rst_n;

    // AXI WRITE
    logic [31:0] awaddr;
    logic awvalid;
    logic awready;

    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic wvalid;
    logic wready;

    logic [1:0] bresp;
    logic bvalid;
    logic bready;

    // AXI READ
    logic [31:0] araddr;
    logic arvalid;
    logic arready;

    logic [31:0] rdata;
    logic [1:0]  rresp;
    logic rvalid;
    logic rready;

    // REG BANK
    logic wr_en;
    logic [3:0] wr_addr;
    logic [31:0] wr_data;

    logic rd_en;
    logic [3:0] rd_addr;

    logic [31:0] rd_data;

    // DUT
    axi_lite_slave dut (

        .s_axi_aclk(clk),
        .s_axi_aresetn(rst_n),

        .s_axi_awaddr(awaddr),
        .s_axi_awvalid(awvalid),
        .s_axi_awready(awready),

        .s_axi_wdata(wdata),
        .s_axi_wstrb(wstrb),
        .s_axi_wvalid(wvalid),
        .s_axi_wready(wready),

        .s_axi_bresp(bresp),
        .s_axi_bvalid(bvalid),
        .s_axi_bready(bready),

        .s_axi_araddr(araddr),
        .s_axi_arvalid(arvalid),
        .s_axi_arready(arready),

        .s_axi_rdata(rdata),
        .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid),
        .s_axi_rready(rready),

        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),

        .rd_en(rd_en),
        .rd_addr(rd_addr),

        .rd_data(rd_data)

    );

    // =========================================================================
    // CLOCK
    // =========================================================================

    initial clk = 0;
    always #5 clk = ~clk;

    // =========================================================================
    // PERFORMANCE
    // =========================================================================

    integer start_cycle;
    integer end_cycle;

    integer cycle_ctr;

    always @(posedge clk or negedge rst_n) begin

        if(!rst_n)
            cycle_ctr <= 0;
        else
            cycle_ctr <= cycle_ctr + 1;

    end

    // =========================================================================
    // READ DATA MODEL
    // =========================================================================

    always_comb begin

        case(rd_addr)

            4'h0: rd_data = 32'h11111111;
            4'h1: rd_data = 32'h22222222;
            4'h2: rd_data = 32'h33333333;

            default:
                rd_data = 32'hDEADBEEF;

        endcase

    end

    // =========================================================================
    // RESET
    // =========================================================================

    task automatic apply_reset();

        begin

            rst_n = 0;

            awaddr  = 0;
            awvalid = 0;

            wdata   = 0;
            wstrb   = 0;
            wvalid  = 0;

            bready  = 0;

            araddr  = 0;
            arvalid = 0;

            rready  = 0;

            repeat(5) @(posedge clk);

            rst_n = 1;

            repeat(5) @(posedge clk);

        end

    endtask

    // =========================================================================
    // AXI WRITE
    // =========================================================================

    task automatic axi_write(

        input [31:0] addr,
        input [31:0] data

    );

        begin

            @(posedge clk);

            awaddr  <= addr;
            awvalid <= 1'b1;

            wdata   <= data;
            wvalid  <= 1'b1;

            wstrb   <= 4'b1111;

            start_cycle = cycle_ctr;

            wait(awready && wready);

            @(posedge clk);

            awvalid <= 0;
            wvalid  <= 0;

            bready <= 1'b1;

            wait(bvalid);

            @(posedge clk);

            bready <= 0;

            end_cycle = cycle_ctr;

            $display("WRITE LATENCY = %0d cycles",
                end_cycle - start_cycle
            );

        end

    endtask

    // =========================================================================
    // AXI READ
    // =========================================================================

    task automatic axi_read(

        input [31:0] addr

    );

        begin

            @(posedge clk);

            araddr  <= addr;
            arvalid <= 1'b1;

            start_cycle = cycle_ctr;

            wait(arready);

            @(posedge clk);

            arvalid <= 0;

            rready <= 1'b1;

            wait(rvalid);

            @(posedge clk);

            rready <= 0;

            end_cycle = cycle_ctr;

            $display(
                "READ addr=%h data=%h latency=%0d cycles",
                addr,
                rdata,
                end_cycle-start_cycle
            );

        end

    endtask

    // =========================================================================
    // TESTS
    // =========================================================================

    task automatic test_basic_write();

        begin

            $display("TEST : BASIC WRITE");

            axi_write(
                32'h00000000,
                32'h12345678
            );

            if(!wr_en) begin

                $display("WRITE FAILURE");
                $finish;
            end

            $display("BASIC WRITE PASSED");

        end

    endtask

    task automatic test_basic_read();

        begin

            $display("TEST : BASIC READ");

            axi_read(32'h00000004);

            if(rdata != 32'h22222222) begin

                $display("READ FAILURE");
                $finish;
            end

            $display("BASIC READ PASSED");

        end

    endtask

    task automatic test_backpressure();

        begin

            $display("TEST : BACKPRESSURE");

            @(posedge clk);

            awaddr  <= 32'h0;
            awvalid <= 1'b1;

            wdata   <= 32'hCAFEBABE;
            wvalid  <= 1'b1;

            wstrb   <= 4'b1111;

            repeat(10)
                @(posedge clk);

            bready <= 1'b1;

            wait(bvalid);

            @(posedge clk);

            awvalid <= 0;
            wvalid  <= 0;
            bready  <= 0;

            $display("BACKPRESSURE PASSED");

        end

    endtask

    task automatic test_bad_strobe();

        begin

            $display("TEST : INVALID STROBE");

            @(posedge clk);

            awaddr  <= 32'h0;
            awvalid <= 1'b1;

            wdata   <= 32'hAAAAAAAA;
            wvalid  <= 1'b1;

            wstrb   <= 4'b0011;

            bready  <= 1'b1;

            wait(bvalid);

            if(bresp != 2'b10) begin

                $display("BAD STROBE FAILURE");
                $finish;
            end

            @(posedge clk);

            awvalid <= 0;
            wvalid  <= 0;
            bready  <= 0;

            $display("INVALID STROBE PASSED");

        end

    endtask

    task automatic test_simultaneous_aw_w();

        begin

            $display("TEST : SIMULTANEOUS AW/W");

            axi_write(
                32'h00000008,
                32'h5555AAAA
            );

            $display("SIMULTANEOUS AW/W PASSED");

        end

    endtask

    task automatic test_stress();

        begin

            $display("TEST : STRESS");

            for(int i=0;i<20;i++) begin

                axi_write(i*4, $random);

                axi_read(i*4);

            end

            $display("STRESS PASSED");

        end

    endtask

    // =========================================================================
    // MAIN
    // =========================================================================

    initial begin

        apply_reset();

        test_basic_write();

        test_basic_read();

        test_backpressure();

        test_bad_strobe();

        test_simultaneous_aw_w();

        test_stress();

        $display("=================================");
        $display("ALL AXI-LITE TESTS PASSED");
        $display("=================================");

        $finish;

    end

endmodule