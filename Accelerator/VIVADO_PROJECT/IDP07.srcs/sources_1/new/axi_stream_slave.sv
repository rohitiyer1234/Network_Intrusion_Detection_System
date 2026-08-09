/* 
`timescale 1ns/1ps

module axi_stream_slave
    import parameter_pkg::*;
#(

    parameter int DATA_WIDTH   = AXIS_W,
    parameter int NUM_FEATURES = L1_IN,
    parameter int FEAT_WIDTH   = FEAT_W

)(

    input logic clk,
    input logic rst_n,

    input logic [DATA_WIDTH-1:0] s_axis_tdata,
    input logic                  s_axis_tvalid,
    output logic                 s_axis_tready,
    input logic                  s_axis_tlast,

    output logic                 fb_write_en,

    output logic signed [FEAT_WIDTH-1:0]
        fb_write_data,

    output logic frame_done,

    input logic fb_full

);

    localparam int CNT_W =
        (NUM_FEATURES <= 1) ? 1 :
        $clog2(NUM_FEATURES);

    logic [CNT_W-1:0] feat_count;

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            s_axis_tready <= 1'b1;

            fb_write_en   <= 1'b0;
            fb_write_data <= '0;

            frame_done <= 1'b0;

            feat_count <= '0;

        end
        else begin

            fb_write_en <= 1'b0;
            frame_done  <= 1'b0;

            // =========================================================
            // Backpressure
            // =========================================================

            s_axis_tready <= !fb_full;

            // =========================================================
            // Accept Feature
            // =========================================================

            if(s_axis_tvalid && s_axis_tready) begin

                fb_write_en <= 1'b1;

                fb_write_data
                    <= $signed(
                        s_axis_tdata[FEAT_WIDTH-1:0]
                    );

                // Last Feature
                if(feat_count == NUM_FEATURES-1) begin

                    frame_done <= 1'b1;

                    feat_count <= '0;

                end
                else begin

                    feat_count <= feat_count + 1'b1;

                end

            end

        end

    end

endmodule
*/

`timescale 1ns/1ps

module axi_stream_slave
    import parameter_pkg::*;
#(

    parameter int DATA_WIDTH   = 8,
    parameter int NUM_FEATURES = L1_IN,
    parameter int FEAT_WIDTH   = FEAT_W

)(

    input logic clk,
    input logic rst_n,

    // =========================================================================
    // AXI STREAM INPUT
    // =========================================================================

    input logic [DATA_WIDTH-1:0] s_axis_tdata,
    input logic                  s_axis_tvalid,
    output logic                 s_axis_tready,
    input logic                  s_axis_tlast,

    // =========================================================================
    // FEATURE BUFFER
    // =========================================================================

    output logic fb_write_en,

    output logic signed [FEAT_WIDTH-1:0]
        fb_write_data,

    output logic frame_done,

    input logic fb_full,

    // =========================================================================
    // STATUS
    // =========================================================================

    output logic packet_error

);

    // =========================================================================
    // COUNTER WIDTH
    // =========================================================================

    localparam int CNT_W =
        (NUM_FEATURES <= 1)
        ? 1
        : $clog2(NUM_FEATURES);

    // =========================================================================
    // INTERNALS
    // =========================================================================

    logic [CNT_W-1:0]
        feat_count;

    // =========================================================================
    // AXI RECEIVE LOGIC
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            s_axis_tready <= 1'b1;

            fb_write_en   <= 1'b0;
            fb_write_data <= '0;

            frame_done   <= 1'b0;
            packet_error <= 1'b0;

            feat_count <= '0;

        end
        else begin

            // =============================================================
            // Defaults
            // =============================================================

            fb_write_en <= 1'b0;

            frame_done <= 1'b0;

            packet_error <= 1'b0;

            // =============================================================
            // Backpressure
            // =============================================================

            s_axis_tready <= !fb_full;

            // =============================================================
            // Accept Beat
            // =============================================================

            if(s_axis_tvalid && s_axis_tready) begin

                fb_write_en <= 1'b1;

                fb_write_data
                    <= $signed(
                        s_axis_tdata[FEAT_WIDTH-1:0]
                    );

                // ---------------------------------------------------------
                // Correct End-of-Frame
                // ---------------------------------------------------------

                if(
                    (feat_count == NUM_FEATURES-1)
                    &&
                    s_axis_tlast
                ) begin

                    frame_done <= 1'b1;

                    feat_count <= '0;

                end

                // ---------------------------------------------------------
                // Early TLAST Error
                // ---------------------------------------------------------

                else if(
                    (feat_count != NUM_FEATURES-1)
                    &&
                    s_axis_tlast
                ) begin

                    packet_error <= 1'b1;

                    feat_count <= '0;

                end

                // ---------------------------------------------------------
                // Missing TLAST Error
                // ---------------------------------------------------------

                else if(
                    (feat_count == NUM_FEATURES-1)
                    &&
                    !s_axis_tlast
                ) begin

                    packet_error <= 1'b1;

                    feat_count <= '0;

                end

                // ---------------------------------------------------------
                // Continue Packet
                // ---------------------------------------------------------

                else begin

                    feat_count <= feat_count + 1'b1;

                end

            end

        end

    end

endmodule