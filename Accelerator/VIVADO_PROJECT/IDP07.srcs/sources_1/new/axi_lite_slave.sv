`timescale 1ns/1ps

module axi_lite_slave
    import parameter_pkg::*;
#(

    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32

)(

    input logic s_axi_aclk,
    input logic s_axi_aresetn,

    // =========================================================================
    // WRITE ADDRESS CHANNEL
    // =========================================================================

    input  logic [ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                  s_axi_awvalid,
    output logic                  s_axi_awready,

    // =========================================================================
    // WRITE DATA CHANNEL
    // =========================================================================

    input  logic [DATA_WIDTH-1:0]   s_axi_wdata,
    input  logic [DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  logic                    s_axi_wvalid,
    output logic                    s_axi_wready,

    // =========================================================================
    // WRITE RESPONSE CHANNEL
    // =========================================================================

    output logic [1:0] s_axi_bresp,
    output logic       s_axi_bvalid,
    input  logic       s_axi_bready,

    // =========================================================================
    // READ ADDRESS CHANNEL
    // =========================================================================

    input  logic [ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                  s_axi_arvalid,
    output logic                  s_axi_arready,

    // =========================================================================
    // READ DATA CHANNEL
    // =========================================================================

    output logic [DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0]            s_axi_rresp,
    output logic                  s_axi_rvalid,
    input  logic                  s_axi_rready,

    // =========================================================================
    // REGISTER BANK INTERFACE
    // =========================================================================

    output logic                  wr_en,
    output logic [3:0]            wr_addr,
    output logic [31:0]           wr_data,

    output logic                  rd_en,
    output logic [3:0]            rd_addr,

    input logic [31:0] rd_data

);

    // =========================================================================
    // WRITE FSM
    // =========================================================================

    typedef enum logic [1:0] {

        WR_IDLE,
        WR_RESP

    } wr_state_t;

    wr_state_t wr_state;

    logic aw_seen;
    logic w_seen;

    logic [ADDR_WIDTH-1:0] awaddr_r;
    logic [DATA_WIDTH-1:0] wdata_r;

    // =========================================================================
    // READ FSM
    // =========================================================================

    typedef enum logic [1:0] {

        RD_IDLE,
        RD_WAIT,
        RD_RESP

    } rd_state_t;

    rd_state_t rd_state;

    logic [31:0] rd_data_r;

    // =========================================================================
    // WRITE FSM
    // =========================================================================

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin

        if(!s_axi_aresetn) begin

            wr_state <= WR_IDLE;

            s_axi_awready <= 1'b1;
            s_axi_wready  <= 1'b1;

            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;

            wr_en   <= 1'b0;
            wr_addr <= '0;
            wr_data <= '0;

            aw_seen <= 1'b0;
            w_seen  <= 1'b0;

            awaddr_r <= '0;
            wdata_r  <= '0;

        end
        else begin

            wr_en <= 1'b0;

            case(wr_state)

                // =========================================================
                // IDLE
                // =========================================================

                WR_IDLE: begin

                    // -----------------------------------------------------
                    // Capture AW
                    // -----------------------------------------------------

                    if(s_axi_awvalid && s_axi_awready) begin

                        awaddr_r <= s_axi_awaddr;

                        aw_seen <= 1'b1;

                        s_axi_awready <= 1'b0;

                    end

                    // -----------------------------------------------------
                    // Capture W
                    // -----------------------------------------------------

                    if(s_axi_wvalid && s_axi_wready) begin

                        wdata_r <= s_axi_wdata;

                        w_seen <= 1'b1;

                        s_axi_wready <= 1'b0;

                    end

                    // -----------------------------------------------------
                    // Write Issue
                    // -----------------------------------------------------

                    if(

                        (aw_seen || (s_axi_awvalid && s_axi_awready))
                        &&
                        (w_seen  || (s_axi_wvalid  && s_axi_wready))

                    ) begin

                        // Full-word writes only
                        if(s_axi_wstrb == 4'b1111) begin

                            wr_en <= 1'b1;

                            wr_addr <=
                                (aw_seen)
                                ? awaddr_r[5:2]
                                : s_axi_awaddr[5:2];

                            wr_data <=
                                (w_seen)
                                ? wdata_r
                                : s_axi_wdata;

                            s_axi_bresp <= 2'b00;

                        end
                        else begin

                            s_axi_bresp <= 2'b10;

                        end

                        s_axi_bvalid <= 1'b1;

                        wr_state <= WR_RESP;

                    end

                end

                // =========================================================
                // RESPONSE
                // =========================================================

                WR_RESP: begin

                    if(s_axi_bvalid && s_axi_bready) begin

                        s_axi_bvalid <= 1'b0;

                        s_axi_awready <= 1'b1;
                        s_axi_wready  <= 1'b1;

                        aw_seen <= 1'b0;
                        w_seen  <= 1'b0;

                        wr_state <= WR_IDLE;

                    end

                end

                default: begin

                    wr_state <= WR_IDLE;

                end

            endcase

        end

    end

    // =========================================================================
    // READ FSM
    // =========================================================================

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin

        if(!s_axi_aresetn) begin

            rd_state <= RD_IDLE;

            s_axi_arready <= 1'b1;

            s_axi_rvalid <= 1'b0;
            s_axi_rresp  <= 2'b00;
            s_axi_rdata  <= '0;

            rd_en   <= 1'b0;
            rd_addr <= '0;

            rd_data_r <= '0;

        end
        else begin

            rd_en <= 1'b0;

            case(rd_state)

                // =========================================================
                // IDLE
                // =========================================================

                RD_IDLE: begin

                    if(s_axi_arvalid && s_axi_arready) begin

                        rd_addr <= s_axi_araddr[5:2];

                        rd_en <= 1'b1;

                        s_axi_arready <= 1'b0;

                        rd_state <= RD_WAIT;

                    end

                end

                // =========================================================
                // WAIT FOR REGISTER BANK
                // =========================================================

                RD_WAIT: begin

                    rd_data_r <= rd_data;

                    s_axi_rresp <= 2'b00;

                    s_axi_rvalid <= 1'b1;

                    rd_state <= RD_RESP;

                end

                // =========================================================
                // RESPONSE
                // =========================================================

                RD_RESP: begin

                    s_axi_rdata <= rd_data_r;

                    if(s_axi_rvalid && s_axi_rready) begin

                        s_axi_rvalid <= 1'b0;

                        s_axi_arready <= 1'b1;

                        rd_state <= RD_IDLE;

                    end

                end

                default: begin

                    rd_state <= RD_IDLE;

                end

            endcase

        end

    end

endmodule