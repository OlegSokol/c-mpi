/*
This file implements the MASTER - WORKERS solution for an embarrasingly parallel problem (i.e. one consisting of multiple independent task of similar complexity).
MPI features used: NEW MPI DATATYPES
*/


#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <utility>
#include <sstream>

#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>

#include "data_reader.h"
#include "data_writer.h"
#include "ucl_2d.h"
#include "fin_instr.h"
#include "calibrate.h"
#include "call_bond_specific.h"

using namespace std;


#define TERM_TAG 1111111111  //termination tag

#define UNIQUE_ID_LEN 9

/* structs used in creation of new MPI datatypes */

typedef struct common_params_st {
    int    m_1, m_2, loc_1, loc_2, max_iter_calibr;
    double lb_v, dv, dt, dt_calibr;
} common_params;



#define NUM_FLT_PER_RECORD 13
#define NUM_INT_PER_RECORD 12

typedef struct input_st {
    char   unique_ID[UNIQUE_ID_LEN+1];
    double lb_r, dr, Market_price, Notional, Recovery, Coupon_rate_annual, Spread, Debt, vol_V, cs_R, lr_R, vol_R, div_y;
    int    Date_trade, Date_issue, Date_first_coupon, Date_maturity, Date_EE_BeginTime, Date_EE_EndTime, DayCount_conv, Coupons_per_annum, hasMakeWhole, has_EE_sched, num_EE_shed_periods, PERMNO;
} input;


typedef struct output_mpi_st {
    char    unique_ID[UNIQUE_ID_LEN+1];
    double  date_trade, market_price_clean, gross_mkt_price, V, vol_V, delta_S_V, delta_S_V_highVol, gamma_S_V,
            delta_Hedge_d1, delta_Hedge_d1_spl, dV_OptFree, dV_OptFree_spl,
            delta_Hedge_d2, delta_Hedge_d2_spl, dR_OptFree, dR_OptFree_spl, dR_DfltFree, dR_DfltFree_spl,
            gamma_1, gamma_2, gamma_1_spl, gamma_2_spl, gamma_1_OptFree, gamma_1_spl_OptFree, gamma_2_OptFree, gamma_2_spl_OptFree,
            delta_1_HV, delta_1_spl_HV, delta_2_HV, delta_2_spl_HV,
            calibr_parameter, barrier_at_trade, hv_price, option_free_price, option_free_hv_price, default_free_price, CIR_price, DD, EDF;
} output_mpi;



/*custom function  for reading from a file
uses a Reader clas for getting individual formatted data records
*/
template <typename T>
input Read_inputs_from_fstream(data_reader<T>& Reader) {
    std::vector<T> record;
    input input_read;
    record = Reader.get_record(input_read.unique_ID);
    input_read.lb_r = record[0];
    input_read.dr = record[1];
    input_read.Market_price = record[2];
    input_read.Notional = record[3];
    input_read.Recovery = record[4];
    input_read.Coupon_rate_annual = record[5];
    input_read.Spread = record[6];
    input_read.Debt = record[7];
    input_read.vol_V = record[8];
    input_read.cs_R = record[9];
    input_read.lr_R = record[10];
    input_read.vol_R = record[11];
    input_read.div_y = record[12];
    input_read.Date_trade = (int) record[13];
    input_read.Date_issue = (int) record[14];
    input_read.Date_first_coupon = (int) record[15];
    input_read.Date_maturity = (int) record[16];
    input_read.Date_EE_BeginTime = (int) record[17];
    input_read.Date_EE_EndTime = (int) record[18];
    input_read.DayCount_conv = (int) record[19];
    input_read.Coupons_per_annum = (int) record[20];
    input_read.hasMakeWhole = (int) record[21];
    input_read.has_EE_sched = (int) record[22];
    input_read.num_EE_shed_periods = (int) record[23];
    input_read.PERMNO = (int) record[24];
    return input_read;
}

/*custom function  for writing to file*/
template <typename T>
void Write_outputs_to_fstream(data_writer<T>& Writer, output_mpi Output) {
    std::vector<T> Record;
    /************ Identifying info ************/
    Record.push_back(Output.date_trade);
    Record.push_back(Output.market_price_clean);
    Record.push_back(Output.gross_mkt_price);
    
    /************* Calibration output *************/
    Record.push_back(Output.V);
    Record.push_back(Output.calibr_parameter);
    //Record.push_back(Output.barrier_at_trade);
    
    /************* MAIN RESULTS *************/
    Record.push_back(Output.delta_Hedge_d1);
    Record.push_back(Output.delta_Hedge_d2);
    Record.push_back(Output.gamma_1);
    Record.push_back(Output.gamma_2);
    
    /************* Option FREE *************/
    Record.push_back(Output.option_free_price);
    Record.push_back(Output.dV_OptFree);
    Record.push_back(Output.dR_OptFree);
    Record.push_back(Output.gamma_1_OptFree);
    Record.push_back(Output.gamma_2_OptFree);
    
    /************* Default FREE *************/
    Record.push_back(Output.default_free_price);
    
    /************* Higher VOL ************/
    Record.push_back(Output.hv_price);
    Record.push_back(Output.option_free_hv_price);
    Record.push_back(Output.delta_1_HV);
    Record.push_back(Output.delta_1_spl_HV);
    
    /************ AUX ************/
    Record.push_back(Output.CIR_price);
    Record.push_back(Output.vol_V);
    Record.push_back(Output.delta_S_V);
    Record.push_back(Output.gamma_S_V);
    Record.push_back(Output.delta_S_V_highVol);
    Record.push_back(Output.DD);
    Record.push_back(Output.EDF);
    
    Writer.put_record(Output.unique_ID, Record);
}






int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    std::cout << argv[0] << " " << argv[1] << " " << atoi(argv[2]) << '\n'; //echo command line arguments - data file and number of parallel tasks comprising the problem
    string path_file_data = string("/Users/olegsokolinskiy/Documents/Research/callable/data/parts/main_") + argv[1] + string(".txt"); //input file on my system
    string path_file_out = string("/Users/olegsokolinskiy/Documents/Research/MPI_frame/out_") + argv[1] + string(".txt"); //output file on my system
    string path_S         = "/Users/olegsokolinskiy/Documents/Research/MPI_frame/indiv_S/";  //path to input files on my system
    string path_EE        = "/Users/olegsokolinskiy/Documents/Research/MPI_frame/indiv_EE/"; //path to input files on my system
    clock_t ini_time = clock();    //timing the run
    int num_cases = atoi(argv[2]); //number of parallel tasks comprising the problem
    common_params cp_Master; //common parameteres for numerical solutions of pricing PDEs
    cp_Master.m_1 = 1000;
    cp_Master.lb_v = 0;
    cp_Master.dv = 0.5;
    cp_Master.loc_1 = 200;
    cp_Master.m_2 = 500;
    cp_Master.loc_2 = 250;
    cp_Master.dt = 0.025;
    cp_Master.dt_calibr = 0.025;
    cp_Master.max_iter_calibr = 25;
    int num_procs, proc_id;
    gsl_set_error_handler_off(); //GSL option
    
    /*************************  REGISTERING NEW MPI DATATYPES  *************************/
    MPI_Datatype MPI_COMMON_PARAMS;
    int   cp_block_counts[2] = {5,4};
    MPI_Datatype cp_types[2] = {MPI_INT, MPI_DOUBLE};
    MPI_Aint   cp_offsets[2];
    cp_offsets[0] = offsetof(common_params, m_1);
    cp_offsets[1] = offsetof(common_params, lb_v);
    MPI_Type_create_struct(2, cp_block_counts, cp_offsets, cp_types, &MPI_COMMON_PARAMS);
    MPI_Type_commit(&MPI_COMMON_PARAMS);
    
    MPI_Datatype MPI_INPUT;
    int   i_block_counts[3] = {UNIQUE_ID_LEN+1, NUM_FLT_PER_RECORD, NUM_INT_PER_RECORD};
    MPI_Datatype i_types[3] = {MPI_CHAR,      MPI_DOUBLE,         MPI_INT};
    MPI_Aint   i_offsets[3];
    i_offsets[0] = offsetof(input, unique_ID);
    i_offsets[1] = offsetof(input, lb_r);
    i_offsets[2] = offsetof(input, Date_trade);
    MPI_Type_create_struct(3, i_block_counts, i_offsets, i_types, &MPI_INPUT);
    MPI_Type_commit(&MPI_INPUT);
    
    MPI_Datatype MPI_OUTPUT;
    int   o_block_counts[2] = {UNIQUE_ID_LEN+1, 39};
    MPI_Datatype o_types[2] = {MPI_CHAR, MPI_DOUBLE};
    MPI_Aint   o_offsets[2];
    o_offsets[0] = offsetof(output_mpi, unique_ID);
    o_offsets[1] = offsetof(output_mpi, date_trade);
    MPI_Type_create_struct(2, o_block_counts, o_offsets, o_types, &MPI_OUTPUT);
    MPI_Type_commit(&MPI_OUTPUT);

    
    /*************************  LAUNCHING MPI  *************************/
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int master_id = 0;
    /*************************  MASTER PROCESS  *************************/
    if (proc_id == master_id) {
        MPI_Status status;
        int num_assigned_cases, num_processed_cases, from, tag;
        MPI_Bcast(&cp_Master, 1, MPI_COMMON_PARAMS, 0, MPI_COMM_WORLD); //broadcasting common parameters
        input input_Master;
        data_reader<double> Reader(path_file_data, NUM_FLT_PER_RECORD+NUM_INT_PER_RECORD, UNIQUE_ID_LEN); //Reader class for input data
        data_writer<double> Writer(path_file_out, UNIQUE_ID_LEN); //Writer class for saving output data
        num_assigned_cases = 0;
        for (int i=1; i < num_procs; ++i) { //initial task assignments
            input_Master = Read_inputs_from_fstream(Reader); //reading input data
            if (num_assigned_cases < num_cases) {
                MPI_Send(&input_Master, 1, MPI_INPUT, i, num_assigned_cases, MPI_COMM_WORLD); //sending data to workers
                num_assigned_cases++; //keeping track of assigned work
            }
            else {
                MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, i, TERM_TAG, MPI_COMM_WORLD); //no more work to be done -> send terminate tag
            }
        }
        output_mpi out_Master;       // place to store output received from worker nodes
        num_processed_cases = 0;
        while (num_processed_cases < num_cases) { //while work remains to be done
            MPI_Recv(&out_Master, 1, MPI_OUTPUT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); //receiving a single unit of custom MPI datatype
            num_processed_cases++;
            from = status.MPI_SOURCE; //sender id
            tag  = status.MPI_TAG; //identifying tag
            std::cout << "Master recevied output " << out_Master.unique_ID << " from worker " << from << " tagged with " << tag << "\n"; //echo the info
            Write_outputs_to_fstream(Writer, out_Master); //save outputs
            if (num_assigned_cases < num_cases) { //if more work remains - send a task to the worker form whom an ouput has been just received
                input_Master = Read_inputs_from_fstream(Reader);
                MPI_Send(&input_Master, 1, MPI_INPUT, from, num_assigned_cases, MPI_COMM_WORLD);
                num_assigned_cases++; //keeping track of assigned work
            }
            else {
                MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, from, TERM_TAG, MPI_COMM_WORLD); //no more work to be done -> send terminate tag
            }
        }
    }
    
    /*************************  WORKER PROCESSES  *************************/
    else {
        MPI_Status status_loc;
        common_params cp_Worker;
        MPI_Bcast(&cp_Worker, 1, MPI_COMMON_PARAMS, 0, MPI_COMM_WORLD); //broadcasting common parameters - receiving
        input        input_Worker;
        output_mpi   output_Worker;
        output       loc_Output, option_free_VALUATION, higher_vol_VALUATION, option_free_higher_vol_VALUATION, default_free_VALUATION;
        double       vol_incr_factor = 1.01, ltr_incr_abs = 0.0125;
        double       calibr_param, delta_S_V, delta_S_V_highVol, gamma_S_V, DD, EDF;
        int          asset_vol_calibr_window = 252;
        string       file_S, file_EE;
        stringstream path_file_S, path_file_EE;
        M_model_parameters params_for_asset_vol_calibr;
        
        while (true) {
            MPI_Recv(&input_Worker, 1, MPI_INPUT, master_id, MPI_ANY_TAG, MPI_COMM_WORLD, &status_loc); //get assignment
            if (status_loc.MPI_TAG == TERM_TAG) {
                printf("No more work to be done by worker node %i\n", proc_id); //no more work to be done
                break;
            }
            else {
                for (int i = 0; i < UNIQUE_ID_LEN+1; ++i) {
                    output_Worker.unique_ID[i] = input_Worker.unique_ID[i];
                }
                output_Worker.date_trade         = input_Worker.Date_trade;
                output_Worker.market_price_clean = input_Worker.Market_price;
                path_file_S << path_S << input_Worker.PERMNO << ".txt";
                file_EE = string(output_Worker.unique_ID)+".txt";
                path_file_EE << path_EE << file_EE;
                params_for_asset_vol_calibr.Maturity = 1.00; //fixed to 1 year for EQUITY valuation
                params_for_asset_vol_calibr.r        = input_Worker.lb_r + cp_Worker.loc_2*input_Worker.dr; //setting up the grid for numerical PDE solution
                params_for_asset_vol_calibr.Vol_V    = input_Worker.vol_V; // initial guess for volatility
                input_Worker.vol_V                   = asset_VOL_estimate(input_Worker.Date_trade, path_file_S.str(), asset_vol_calibr_window, &params_for_asset_vol_calibr, vol_incr_factor, &delta_S_V, &delta_S_V_highVol, &gamma_S_V, &DD, &EDF); //Estimating asset volatility
                output_Worker.vol_V                  = input_Worker.vol_V;
                output_Worker.delta_S_V              = delta_S_V;
                output_Worker.DD                     = DD;
                output_Worker.EDF                    = EDF;
                output_Worker.delta_S_V_highVol      = delta_S_V_highVol;
                output_Worker.gamma_S_V              = gamma_S_V;
                
                /* THIS IS WHERE THE MAIN WORK IS DONE */
                Calibrate_CALLABLE_BOND(true, output_Worker.unique_ID, cp_Worker.max_iter_calibr,
                                              input_Worker.Notional, input_Worker.Recovery,
                                              input_Worker.Date_trade, input_Worker.Date_issue, input_Worker.Date_first_coupon, input_Worker.Date_maturity,
                                              input_Worker.DayCount_conv,
                                              input_Worker.Coupon_rate_annual, input_Worker.Coupons_per_annum,
                                              input_Worker.has_EE_sched, input_Worker.num_EE_shed_periods, path_file_EE.str(),
                                              input_Worker.hasMakeWhole, input_Worker.Date_EE_BeginTime, input_Worker.Date_EE_EndTime, input_Worker.Spread,
                                              input_Worker.Market_price, input_Worker.vol_V, input_Worker.cs_R, input_Worker.lr_R, input_Worker.vol_R, input_Worker.div_y,
                                                     cp_Worker.m_1, cp_Worker.m_2, cp_Worker.loc_1, cp_Worker.loc_2,  cp_Worker.lb_v, input_Worker.lb_r, cp_Worker.dv, input_Worker.dr, cp_Worker.dt, cp_Worker.dt_calibr,
                                                            &loc_Output, &higher_vol_VALUATION, &option_free_VALUATION, &option_free_higher_vol_VALUATION, &default_free_VALUATION, &calibr_param, ltr_incr_abs, vol_incr_factor);
                
                /* SAVING OUTPUT */
                output_Worker.gross_mkt_price    = loc_Output.gross_mkt_price;
                output_Worker.V                  = loc_Output.V;
                output_Worker.calibr_parameter   = calibr_param;
                output_Worker.barrier_at_trade   = calibr_param*exp(-(input_Worker.lr_R-input_Worker.div_y)*(input_Worker.Date_maturity-input_Worker.Date_trade)/365);
                output_Worker.delta_Hedge_d1     = loc_Output.delta_1;
                output_Worker.delta_Hedge_d2     = loc_Output.delta_2;
                output_Worker.delta_Hedge_d1_spl = loc_Output.delta_1_interpol;
                output_Worker.delta_Hedge_d2_spl = loc_Output.delta_2_interpol;
                output_Worker.option_free_price  = option_free_VALUATION.V;
                output_Worker.dV_OptFree         = option_free_VALUATION.delta_1; output_Worker.dV_OptFree_spl   = option_free_VALUATION.delta_1_interpol;
                output_Worker.dR_OptFree         = option_free_VALUATION.delta_2; output_Worker.dR_OptFree_spl   = option_free_VALUATION.delta_2_interpol;
                output_Worker.default_free_price = default_free_VALUATION.V;
                output_Worker.dR_DfltFree        = default_free_VALUATION.delta_2;  output_Worker.dR_DfltFree_spl  = default_free_VALUATION.delta_2_interpol;
                output_Worker.CIR_price          = loc_Output.analytical_noOpt_price;
                output_Worker.gamma_1     = loc_Output.gamma_1;
                output_Worker.gamma_2     = loc_Output.gamma_2;
                output_Worker.gamma_1_spl = loc_Output.gamma_1_interpol;
                output_Worker.gamma_2_spl = loc_Output.gamma_2_interpol;
                output_Worker.gamma_1_OptFree     = option_free_VALUATION.gamma_1;
                output_Worker.gamma_2_OptFree     = option_free_VALUATION.gamma_2;
                output_Worker.gamma_1_spl_OptFree = option_free_VALUATION.gamma_1_interpol;
                output_Worker.gamma_2_spl_OptFree = option_free_VALUATION.gamma_2_interpol;
                output_Worker.delta_1_HV     = higher_vol_VALUATION.delta_1;
                output_Worker.delta_1_spl_HV = higher_vol_VALUATION.delta_1_interpol;
                output_Worker.delta_2_HV     = higher_vol_VALUATION.delta_2;
                output_Worker.delta_2_spl_HV = higher_vol_VALUATION.delta_2_interpol;
                output_Worker.hv_price       = higher_vol_VALUATION.V;
                output_Worker.option_free_hv_price = option_free_higher_vol_VALUATION.V;

                MPI_Send(&output_Worker, 1, MPI_OUTPUT, master_id, proc_id, MPI_COMM_WORLD);
                
                path_file_S.str(string());
                path_file_EE.str(string());
            }
        }
        
    }
    
    double time_elapsed = (double(clock()-ini_time)/CLOCKS_PER_SEC)/3600;
    if (proc_id == master_id) {
        std::cerr << "Time elapsed: " << time_elapsed << " hours. Average time per bond: " << time_elapsed/num_cases << " hours\n";
    }

    MPI_Finalize();
}