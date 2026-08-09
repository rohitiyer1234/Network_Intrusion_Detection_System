/*
`timescale 1ns/1ps

module register_bank
  import parameter_pkg::*;
#(
  parameter int NUM_FEATURES = 20,
  parameter int NUM_CLASSES  = 3
)(
  input  logic clk,
  input  logic rst_n,

  // ============================================================
  // WRITE PORT
  // ============================================================
  input  logic        wr_en,
  input  logic [3:0]  wr_addr,
  input  logic [31:0] wr_data,

  // ============================================================
  // READ PORT
  // ============================================================
  input  logic        rd_en,
  input  logic [3:0]  rd_addr,
  output logic [31:0] rd_data,

  // ============================================================
  // CONTROL OUTPUTS
  // ============================================================
  output logic ctrl_start,
  output logic ctrl_reset,

  // ============================================================
  // STATUS INPUTS
  // ============================================================
  input logic status_idle,
  input logic status_busy,
  input logic status_done,

  input logic [31:0] inf_counter,
  input logic [31:0] latency_cycles,

  // ============================================================
  // LOGITS
  // ============================================================
  input logic logit_we,

  input logic signed [ACC_W-1:0]
      logits [NUM_CLASSES-1:0]
);

  // ===========================================================================
  // CONTROL REGISTER
  // ===========================================================================
  logic [31:0] control_reg;

  // ===========================================================================
  // LOGIT STORAGE
  // ===========================================================================
  logic signed [31:0] logit_regs [NUM_CLASSES-1:0];

  // ===========================================================================
  // WRITE LOGIC
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin

    if (!rst_n) begin

      control_reg <= 0;

      ctrl_start <= 0;
      ctrl_reset <= 0;

      for (int i = 0; i < NUM_CLASSES; i++)
        logit_regs[i] <= 0;

    end else begin

      ctrl_start <= 1'b0;
      ctrl_reset <= 1'b0;

      // ----------------------------------------------------------
      // STORE LOGITS
      // ----------------------------------------------------------
      if (logit_we) begin

        for (int i = 0; i < NUM_CLASSES; i++)
          logit_regs[i] <=
              32'(signed'(logits[i]));

      end

      // ----------------------------------------------------------
      // CPU WRITE
      // ----------------------------------------------------------
      if (wr_en) begin

        case (wr_addr)

          // CONTROL
          4'h0: begin

            control_reg <= wr_data;

            ctrl_start <= wr_data[0] && !status_busy;
            ctrl_reset <= wr_data[1];

          end

          default: begin
          end

        endcase

      end
      
      if(ctrl_start)
            control_reg[0] <= 1'b0;

      if(ctrl_reset)
            control_reg[1] <= 1'b0;
    end

  end

  // ===========================================================================
  // READ MUX
  // ===========================================================================
  always_ff @(posedge clk or negedge rst_n) begin

    if (!rst_n) begin

      rd_data <= 0;

    end else begin

      if (rd_en) begin

        case (rd_addr)

          // CONTROL
          4'h0:
            rd_data <= control_reg;

          // STATUS
          4'h1:
            rd_data <= {
                29'b0,
                status_done,
                status_busy,
                status_idle
            };

          // NUM_FEATURES
          4'h2:
            rd_data <= NUM_FEATURES;

          // NUM_CLASSES
          4'h3:
            rd_data <= NUM_CLASSES;

          // INFERENCE COUNTER
          4'h4:
            rd_data <= inf_counter;

          // LATENCY
          4'h5:
            rd_data <= latency_cycles;

          // LOGITS
          4'h8:
            rd_data <= logit_regs[0];

          4'h9:
            rd_data <= logit_regs[1];

          4'hA:
            rd_data <= logit_regs[2];

          default:
            rd_data <= 32'hDEADBEEF;

        endcase

      end

    end

  end

endmodule
*/

`timescale 1ns/1ps

module register_bank
    import parameter_pkg::*;
#(

    parameter int NUM_FEATURES = 20,
    parameter int NUM_CLASSES  = 3

)(

    input logic clk,
    input logic rst_n,

    // =========================================================================
    // WRITE PORT
    // =========================================================================

    input logic        wr_en,
    input logic [3:0]  wr_addr,
    input logic [31:0] wr_data,

    // =========================================================================
    // READ PORT
    // =========================================================================

    input logic       rd_en,
    input logic [3:0] rd_addr,

    output logic [31:0] rd_data,

    // =========================================================================
    // CONTROL OUTPUTS
    // =========================================================================

    output logic ctrl_start,
    output logic ctrl_reset,

    // =========================================================================
    // STATUS INPUTS
    // =========================================================================

    input logic status_idle,
    input logic status_busy,
    input logic status_done,

    input logic [31:0] inf_counter,
    input logic [31:0] latency_cycles,

    // =========================================================================
    // LOGITS
    // =========================================================================

    input logic logit_we,

    input logic signed [ACC_W-1:0]
        logits [NUM_CLASSES-1:0]

);

    // =========================================================================
    // REGISTERS
    // =========================================================================

    logic [31:0] control_reg;

    logic signed [31:0]
        logit_regs [NUM_CLASSES-1:0];

    // =========================================================================
    // WRITE LOGIC
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            control_reg <= '0;

            ctrl_start <= 1'b0;
            ctrl_reset <= 1'b0;

            for(int i=0;i<NUM_CLASSES;i++)
                logit_regs[i] <= '0;

        end
        else begin

            // =============================================================
            // Default Pulses
            // =============================================================

            ctrl_start <= 1'b0;
            ctrl_reset <= 1'b0;

            // =============================================================
            // Store Logits
            // =============================================================

            if(logit_we) begin

                for(int i=0;i<NUM_CLASSES;i++) begin

                    logit_regs[i] <= {

                        {(32-ACC_W)
                            {logits[i][ACC_W-1]}},

                        logits[i]

                    };

                end

            end

            // =============================================================
            // CPU WRITE
            // =============================================================

            if(wr_en) begin

                case(wr_addr)

                    // -----------------------------------------------------
                    // CONTROL
                    // -----------------------------------------------------

                    4'h0: begin

                        // Store only persistent bits
                        control_reg <= {

                            wr_data[31:2],
                            2'b00

                        };

                        // Pulse outputs
                        ctrl_start <=
                            wr_data[0]
                            &&
                            !status_busy;

                        ctrl_reset <=
                            wr_data[1];

                    end

                    default: begin
                    end

                endcase

            end

        end

    end

    // =========================================================================
    // READ LOGIC
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if(!rst_n) begin

            rd_data <= '0;

        end
        else begin

            if(rd_en) begin

                case(rd_addr)

                    // -----------------------------------------------------
                    // CONTROL
                    // -----------------------------------------------------

                    4'h0:
                        rd_data <= control_reg;

                    // -----------------------------------------------------
                    // STATUS
                    // -----------------------------------------------------

                    4'h1:
                        rd_data <= {

                            29'b0,

                            status_done,
                            status_busy,
                            status_idle

                        };

                    // -----------------------------------------------------
                    // FEATURES
                    // -----------------------------------------------------

                    4'h2:
                        rd_data <= NUM_FEATURES;

                    // -----------------------------------------------------
                    // CLASSES
                    // -----------------------------------------------------

                    4'h3:
                        rd_data <= NUM_CLASSES;

                    // -----------------------------------------------------
                    // COUNTER
                    // -----------------------------------------------------

                    4'h4:
                        rd_data <= inf_counter;

                    // -----------------------------------------------------
                    // LATENCY
                    // -----------------------------------------------------

                    4'h5:
                        rd_data <= latency_cycles;

                    // -----------------------------------------------------
                    // LOGITS
                    // -----------------------------------------------------

                    4'h8:
                        rd_data <= logit_regs[0];

                    4'h9:
                        rd_data <= logit_regs[1];

                    4'hA:
                        rd_data <= logit_regs[2];

                    // -----------------------------------------------------
                    // INVALID
                    // -----------------------------------------------------

                    default:
                        rd_data <= 32'hDEADBEEF;

                endcase

            end
            else begin

                rd_data <= 32'h00000000;

            end

        end

    end

endmodule