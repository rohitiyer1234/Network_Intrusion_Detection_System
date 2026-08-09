`timescale 1ns/1ps

module feature_buffer
    import parameter_pkg::*;
#(

    parameter int NUM_FEATURES = 20,
    parameter int DATA_WIDTH   = 8

)(

    input logic clk,
    input logic rst_n,

    input logic clear,

    input logic write_en,

    input logic signed [DATA_WIDTH-1:0]
        write_data,

    output logic signed [DATA_WIDTH-1:0]
        features [NUM_FEATURES-1:0],

    output logic full

);

    // =========================================================================
    // Pointer Width
    // =========================================================================

    localparam int PTR_W =
        (NUM_FEATURES <= 1) ? 1 :
        $clog2(NUM_FEATURES);

    // =========================================================================
    // Internal
    // =========================================================================

    logic [PTR_W-1:0] wr_ptr;

    // =========================================================================
    // Sequential Logic
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            wr_ptr <= '0;

            full <= 1'b0;

            for(int i=0;i<NUM_FEATURES;i++)
                features[i] <= '0;

        end
        else begin

            // =============================================================
            // CLEAR
            // =============================================================

            if(clear) begin

                wr_ptr <= '0;

                full <= 1'b0;

            end

            // =============================================================
            // WRITE
            // =============================================================

            else if(write_en && !full) begin

                // Safety against pointer corruption
                if(wr_ptr < NUM_FEATURES) begin

                    features[wr_ptr]
                        <= write_data;

                    // Last feature received
                    if(wr_ptr == (NUM_FEATURES-1)) begin

                        full <= 1'b1;

                    end
                    else begin

                        wr_ptr <= wr_ptr + 1'b1;

                    end

                end
                else begin

                    // Illegal state recovery
                    wr_ptr <= '0;

                    full <= 1'b0;

                end

            end

        end

    end

endmodule