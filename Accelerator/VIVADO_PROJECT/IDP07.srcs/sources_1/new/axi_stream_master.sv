// =============================================================================
// axi_stream_master.sv
// AXI4-Stream master for 3-class MLP output logits
//
// Updated for:
//   - 3 logits only
//   - Single 64-bit beat
//   - Timing-safe AXI valid handling
//   - Registered outputs
//   - No stale beat hazards
//   - Synthesizable Vivado-safe RTL
//
// Beat packing:
//   { logits[1], logits[0] }
//
// logits[2] is sent in upper 32 bits of second beat.
//
// Beat 0:
//   [63:32] = logits[1]
//   [31:0 ] = logits[0]
//
// Beat 1:
//   [63:32] = 32'h0
//   [31:0 ] = logits[2]
//
// tlast asserted on beat 1
// =============================================================================
/*
`timescale 1ns/1ps

module axi_stream_master
  import parameter_pkg::*;
#(
  parameter int DATA_WIDTH  = AXIS_W,
  parameter int NUM_CLASSES = L3_OUT
)(
  input  logic clk,
  input  logic rst_n,

  input  logic send_req,

  input  logic signed [ACC_W-1:0]
      logits [NUM_CLASSES-1:0],

  output logic [DATA_WIDTH-1:0] m_axis_tdata,
  output logic                  m_axis_tvalid,
  input  logic                  m_axis_tready,
  output logic                  m_axis_tlast,

  output logic                  busy
);

  typedef enum logic [1:0] {
    S_IDLE,
    S_SEND0,
    S_SEND1
  } state_t;

  state_t state;

  logic [63:0] beat0;
  logic [63:0] beat1;

  // ===========================================================================
  // FSM
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin

    if (!rst_n) begin

      state <= S_IDLE;

      busy <= 0;

      m_axis_tdata  <= 0;
      m_axis_tvalid <= 0;
      m_axis_tlast  <= 0;

      beat0 <= 0;
      beat1 <= 0;

    end else begin

      case (state)

        // ==============================================================
        S_IDLE: begin

          m_axis_tvalid <= 0;
          m_axis_tlast  <= 0;

          busy <= 0;

          if(send_req && !busy) begin

            // ----------------------------------------------------------
            // PACK DATA
            // ----------------------------------------------------------
            beat0 <= {
                32'(signed'(logits[1])),
                32'(signed'(logits[0]))
            };

            beat1 <= {
                32'h00000000,
                32'(signed'(logits[2]))
            };

            // ----------------------------------------------------------
            // PRESENT FIRST BEAT
            // ----------------------------------------------------------
            m_axis_tdata <= {
                32'(signed'(logits[1])),
                32'(signed'(logits[0]))
            };

            m_axis_tvalid <= 1'b1;
            m_axis_tlast  <= 1'b0;

            busy <= 1'b1;

            state <= S_SEND0;

          end

        end

        // ==============================================================
        S_SEND0: begin

          if (m_axis_tvalid && m_axis_tready) begin

            m_axis_tdata <= beat1;

            m_axis_tvalid <= 1'b1;
            m_axis_tlast  <= 1'b1;

            state <= S_SEND1;

          end

        end

        // ==============================================================
        S_SEND1: begin

          if (m_axis_tvalid && m_axis_tready) begin

            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;

            busy <= 1'b0;

            state <= S_IDLE;

          end

        end

      endcase

    end

  end

endmodule

*/
`timescale 1ns/1ps

module axi_stream_master
    import parameter_pkg::*;
#(

    parameter int DATA_WIDTH  = AXIS_W,
    parameter int NUM_CLASSES = L3_OUT

)(

    input logic clk,
    input logic rst_n,

    input logic send_req,

    input logic signed [ACC_W-1:0]
        logits [NUM_CLASSES-1:0],

    output logic [DATA_WIDTH-1:0]
        m_axis_tdata,

    output logic m_axis_tvalid,

    input logic m_axis_tready,

    output logic m_axis_tlast,

    output logic busy,

    output logic tx_done

);

    // =========================================================================
    // FSM
    // =========================================================================

    typedef enum logic [1:0] {

        S_IDLE,
        S_SEND0,
        S_SEND1

    } state_t;

    state_t state;

    // =========================================================================
    // Sequential
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            state <= S_IDLE;

            m_axis_tdata  <= '0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;

            busy <= 1'b0;

            tx_done <= 1'b0;

        end
        else begin

            // =============================================================
            // Default Pulses
            // =============================================================

            tx_done <= 1'b0;

            case(state)

                // =========================================================
                // IDLE
                // =========================================================

                S_IDLE: begin

                    busy <= 1'b0;

                    m_axis_tvalid <= 1'b0;

                    m_axis_tlast <= 1'b0;

                    // -----------------------------------------------------
                    // Start Transfer
                    // -----------------------------------------------------

                    if(send_req) begin

                        busy <= 1'b1;

                        m_axis_tdata <= {

                            32'(signed'(logits[1])),
                            32'(signed'(logits[0]))

                        };

                        m_axis_tvalid <= 1'b1;

                        m_axis_tlast <= 1'b0;

                        state <= S_SEND0;

                    end

                end

                // =========================================================
                // SEND BEAT0
                // =========================================================

                S_SEND0: begin

                    // Hold outputs stable during stall
                    if(m_axis_tvalid && m_axis_tready) begin

                        m_axis_tdata <= {

                            32'h00000000,
                            32'(signed'(logits[2]))

                        };

                        m_axis_tvalid <= 1'b1;

                        m_axis_tlast <= 1'b1;

                        state <= S_SEND1;

                    end

                end

                // =========================================================
                // SEND BEAT1
                // =========================================================

                S_SEND1: begin

                    // Hold outputs stable during stall
                    if(m_axis_tvalid && m_axis_tready) begin

                        m_axis_tvalid <= 1'b0;

                        m_axis_tlast <= 1'b0;

                        busy <= 1'b0;

                        tx_done <= 1'b1;

                        state <= S_IDLE;

                    end

                end

                // =========================================================
                // Recovery
                // =========================================================

                default: begin

                    state <= S_IDLE;

                    busy <= 1'b0;

                    m_axis_tvalid <= 1'b0;

                    m_axis_tlast <= 1'b0;

                end

            endcase

        end

    end

endmodule