//  adi.h
//  PDE_solver
//  Created by OLEG SOKOLINSKIY on 9/17/16.
//  Copyright © 2016 OLEG SOKOLINSKIY. All rights reserved.


/*
IMPLEMENTS AN ADI ALGORITHM FOR SOLVING BIVARIATE BLACK-SCHOLES MERTON PRICING PARTIAL DIFFERENTIAL EQUATIONS
*/


#ifndef adi_h
#define adi_h

#include <gsl/gsl_linalg.h>

#include <gsl/gsl_spline.h>


#include "ucl_2d.h"
#include "fin_instr.h"


//class for handling time discretization - includes coupon dates into the time grid and reports the corresponding time increments
class time_discr_unif {
public:
    double time_increment(double time_instant, fin_instrument * Fin_instrument) {
        double preceeding_c_time = 0.0;
        double dt = 0.0;
        if (time_instant - Fin_instrument->current_time <= 0) {
            return 0;
        }
        if (Fin_instrument->has_Coupon == true) {
            
            double frst_cpn_time_after_trade = Fin_instrument->Coupon_info->following_coupon_time(Fin_instrument->current_time);
            
            if (time_instant < frst_cpn_time_after_trade) {
                dt = (time_instant - Fin_instrument->current_time > delta_t) ? delta_t : (time_instant - Fin_instrument->current_time);
            }
            else {
                preceeding_c_time = Fin_instrument->Coupon_info->preceeding_coupon_time(time_instant);
                if (time_instant == preceeding_c_time) {
                    dt = delta_t;
                }
                else {
                    dt = (time_instant - preceeding_c_time >= delta_t) ? delta_t : (time_instant - preceeding_c_time);
                }
                
            }
        }
        else
            dt = (time_instant - Fin_instrument->current_time > delta_t) ? delta_t : (time_instant - Fin_instrument->current_time);
        return dt;
    }
    
    double delta_t;
    int T;          // supports time discretization with a fixed step and a fixed number of total steps to maturity
    
};



struct output {
    double gross_mkt_price, analytical_noOpt_price; //dirty price, hypothetical price if the bond were not callable
    double V;
    double delta_1, delta_2, gamma_1, gamma_2; //Greeks
    double delta_1_interpol, delta_2_interpol, gamma_1_interpol, gamma_2_interpol;
};






/*
class for solving numerical PDEs using the ADI method:
includes functionality for bond coupons and callable bonds and make-whole provision
*/
class ADI {
    
public:
    ADI(L_operator * L_, time_discr_unif Time_discr_, fin_instrument * Fin_instrument_)
    :L(L_), Time_discr(Time_discr_), Fin_instrument(Fin_instrument_)
    {
        STATUS      = -100;
        V_2d_payoff = gsl_matrix_alloc(L->St_sp.m_1, L->St_sp.m_2);
        for (int j=0; j < L->St_sp.m_2; j++) {
            for (int i=0; i < L->St_sp.m_1; i++) {
                gsl_matrix_set(V_2d_payoff, i, j, Fin_instrument->payoff->eval(L->St_sp.x_lb+i*L->St_sp.d_x));    // this is for payoffs that vary in only one dimension
            }
        }
    }
    
    ~ADI()
    {
        gsl_matrix_free(V_2d_payoff);
        if(STATUS != -100) gsl_matrix_free(V_0); //an attempt to calculate V_0 has been made
    }
    
private:
    time_discr_unif Time_discr;
    fin_instrument * Fin_instrument; //pointer to the financial instrument
    gsl_matrix * V_2d_payoff;
    
public:
    L_operator * L;
    int STATUS;
    gsl_matrix * V_0;
    
    //component of ADI method - input: direction
    int adi_direction(int direction, double d_t, gsl_matrix * V_2d_in, gsl_matrix * V_2d_out)
    {
        int m, n, status = -1;
        std::vector<lcu> * Lcu_LHS, * Lcu_RHS;
        if(direction == 1) {                      // ADI DIRECTION = 1
            m = L->St_sp.m_1;
            n = L->St_sp.m_2;
            Lcu_LHS = &L->DIAG_L_1_m2;
            Lcu_RHS = &L->DIAG_L_2_m2;
        }
        else {
            m = L->St_sp.m_2;
            n = L->St_sp.m_1;
            
            Lcu_LHS = &L->DIAG_L_2_m1;
            Lcu_RHS = &L->DIAG_L_1_m1;
        }
        lcu Lcu_direc, Lcu_non_direc;
        Lcu_direc.l = gsl_vector_alloc((*Lcu_LHS)[0].l->size);
        Lcu_direc.c = gsl_vector_alloc((*Lcu_LHS)[0].c->size);
        Lcu_direc.u = gsl_vector_alloc((*Lcu_LHS)[0].u->size);
        Lcu_non_direc.l = gsl_vector_alloc((*Lcu_RHS)[0].l->size);
        Lcu_non_direc.c = gsl_vector_alloc((*Lcu_RHS)[0].c->size);
        Lcu_non_direc.u = gsl_vector_alloc((*Lcu_RHS)[0].u->size);
        
        gsl_vector * M = gsl_vector_calloc(m);
        for (int j=0; j<n; j++) {                                  //iteration over non-direction state y (AP: direction 1 -> j_{2} - fixed)
            gsl_vector_memcpy(Lcu_direc.l, (*Lcu_LHS)[j].l);       //IMPORTANT: memcpy PREVENTS the scaling and add operations from being applied multiple times to the same vectors
            gsl_vector_memcpy(Lcu_direc.c, (*Lcu_LHS)[j].c);
            gsl_vector_memcpy(Lcu_direc.u, (*Lcu_LHS)[j].u);
            gsl_vector_memcpy(Lcu_non_direc.l, (*Lcu_RHS)[j].l);
            gsl_vector_memcpy(Lcu_non_direc.c, (*Lcu_RHS)[j].c);
            gsl_vector_memcpy(Lcu_non_direc.u, (*Lcu_RHS)[j].u);

            //SCALING OF DIAGONALS - has to be done here for the GENERAL CASE of NON-EQUIDISTANT TIME GRIDS
            gsl_vector_scale( Lcu_direc.u, -d_t/2);  // LHS of the equation for a given (direction 1 -> j_{2})
            gsl_vector_scale( Lcu_direc.l, -d_t/2);
            gsl_vector_scale( Lcu_direc.c, -d_t/2);
            gsl_vector_add_constant(Lcu_direc.c, 1.00);

            gsl_vector_scale( Lcu_non_direc.u, d_t/2);   // RHS of the equation for a given (direction 1 -> j_{2})
            gsl_vector_scale( Lcu_non_direc.l, d_t/2);
            gsl_vector_scale( Lcu_non_direc.c, d_t/2);
            gsl_vector_add_constant(Lcu_non_direc.c, 1.00);
            
            if (j == 0) {
                for (int i=0; i < m; i++) {
                    gsl_vector_set(M, i,    gsl_vector_get(Lcu_non_direc.c, i)*gsl_matrix_get(V_2d_in, i, j)   + gsl_vector_get(Lcu_non_direc.u, i)*gsl_matrix_get(V_2d_in, i, j+1)  );
                }
            }
            
            else if (j == n-1) {
                for (int i=0; i < m; i++) {
                    gsl_vector_set(M, i,    gsl_vector_get(Lcu_non_direc.l, i)*gsl_matrix_get(V_2d_in, i, j-1) + gsl_vector_get(Lcu_non_direc.c, i)*gsl_matrix_get(V_2d_in, i, j)    );
                }
            }
            
            else {
                for (int i=0; i < m; i++) {
                    gsl_vector_set(M, i,      gsl_vector_get(Lcu_non_direc.l, i)*gsl_matrix_get(V_2d_in, i, j-1) + gsl_vector_get(Lcu_non_direc.c, i)*gsl_matrix_get(V_2d_in, i, j) + gsl_vector_get(Lcu_non_direc.u, i)*gsl_matrix_get(V_2d_in, i, j+1)   );
                }
            }
            
            //SOLVING THE SYSTEM (solution goes into the j^th row)
            gsl_vector_view vec = gsl_matrix_row(V_2d_out,j);
            status = gsl_linalg_solve_tridiag(Lcu_direc.c, Lcu_direc.u, Lcu_direc.l, M, &vec.vector);
            if (status != 0) status = -1;
            
        }
        gsl_vector_free(M); // FREE allocated vectors
        gsl_vector_free(Lcu_direc.l);
        gsl_vector_free(Lcu_direc.c);
        gsl_vector_free(Lcu_direc.u);
        gsl_vector_free(Lcu_non_direc.l);
        gsl_vector_free(Lcu_non_direc.c);
        gsl_vector_free(Lcu_non_direc.u);
        return status;
    }
    
    
    
    //accounts for early exercise (make whole provision and early exercise schedule - callable bonds) and coupon
    void value_adjust_at_time_period_boundaries(double time_instant, gsl_matrix * V) {
        bool exercisable;
        //EARLY EXERCISE
        if (Fin_instrument->has_Make_Whole == true) {
            exercisable = Fin_instrument->MWH_Schedule->make_whole_call_apply(time_instant, V, &(L->St_sp), Fin_instrument->maturity-time_instant, L->Model);
            //std::cerr << "adi.h: At time t " << time_instant << " the bond's MakeWhole is: " << exercisable << '\n';
        }
        //EARLY EXERCISE
        if (Fin_instrument->has_EE_Schedule == true) {
            exercisable = Fin_instrument->EE_Schedule->eval(time_instant, V, &(L->St_sp), Fin_instrument->maturity-time_instant, L->Model);
            //std::cerr << "adi.h: At time t " << time_instant << " the bond's EE status is: " << exercisable << " and the MIN_flag: " << Fin_instrument->EE_Schedule->MIN_flag << '\n';
        }
        if (Fin_instrument->has_Coupon == true) {
            double C = Fin_instrument->Coupon_info->eval(time_instant);
            gsl_matrix_add_constant(V, C);
        }
        //BARRIER: barrier check - AFTER the addition of the coupon (otherwise bondholder would have gotten recovery + coupon) - in all states where the issuer defaults, the asset value will be set to Recovery
        if (Fin_instrument->has_Barrier == true) {
            Fin_instrument->Barrier->eval(time_instant, Fin_instrument->maturity, V, L->St_sp);
        }

    }
    
    
    void ADI_execute() {
        int status_1 = -1, status_2 = -1;
        gsl_matrix * U = gsl_matrix_alloc(L->St_sp.m_2, L->St_sp.m_1);
        gsl_matrix * V = gsl_matrix_alloc(L->St_sp.m_1, L->St_sp.m_2);
        double time_instant, dt;
        time_instant = Fin_instrument->maturity;
        gsl_matrix_memcpy(V, V_2d_payoff);
        value_adjust_at_time_period_boundaries(time_instant, V);

        while(time_instant > Fin_instrument->current_time) {
            dt = Time_discr.time_increment(time_instant, Fin_instrument);
            time_instant = time_instant - dt;
            status_1 = adi_direction(1, dt, V, U);
            status_2 = adi_direction(2, dt, U, V);
            STATUS = GSL_MIN(status_1, status_2);
            value_adjust_at_time_period_boundaries(time_instant, V);
        }
        V_0 = gsl_matrix_alloc(L->St_sp.m_1, L->St_sp.m_2);
        STATUS = -1;                                          //registering the allocation of V_0 for use in DESTRUCTOR - STATUS is no longer -100
        gsl_matrix_memcpy(V_0, V);
        gsl_matrix_free(U);
        gsl_matrix_free(V);
        STATUS = GSL_MIN(status_1, status_2);
    }
    
    
    //generating output
    output Output(int loc_1 = -1, int loc_2 = -1)
    {
        output Out;
        int c_1 = loc_1, c_2 = loc_2;
        if (loc_1 == -1 | loc_2 == -1) {
             c_1 = int(L->St_sp.m_1/2), c_2 = int(L->St_sp.m_2/2);
        }
        Out.V = gsl_matrix_get(V_0, c_1, c_2);
        int incr_r = int(0.0005/L->St_sp.d_y);
        /****************************************   CALCULATING GREEKS   ****************************************/
        Out.delta_1 = (gsl_matrix_get(V_0, c_1+1, c_2)-gsl_matrix_get(V_0, c_1-1, c_2))/(2*L->St_sp.d_x);
        Out.delta_2 = (gsl_matrix_get(V_0, c_1, c_2+incr_r)-gsl_matrix_get(V_0, c_1, c_2-incr_r))/(2*0.0005);
        Out.gamma_1 = (gsl_matrix_get(V_0, c_1+1, c_2)-2*gsl_matrix_get(V_0, c_1, c_2)+gsl_matrix_get(V_0, c_1-1, c_2))/ gsl_pow_2(L->St_sp.d_x);
        Out.gamma_2 = (gsl_matrix_get(V_0, c_1, c_2+1)-2*gsl_matrix_get(V_0, c_1, c_2)+gsl_matrix_get(V_0, c_1, c_2-1))/ gsl_pow_2(L->St_sp.d_y);
        /****************************************   CALCULATING GREEKS BASED ON INTERPOLATION  ****************************************/
        gsl_vector_view center_column = gsl_matrix_column(V_0, c_2);
        gsl_vector_view center_row    = gsl_matrix_row(V_0, c_1);
        const int m_1 = L->St_sp.m_1;
        const int m_2 = L->St_sp.m_2;
        double x_c[m_1], y_r[m_2];
        double axis_x[m_1], axis_y[m_2];
        for (int i = 0; i<L->St_sp.m_1; i++) {
            x_c[i]    = gsl_vector_get(&center_column.vector, i);
            axis_x[i] = L->St_sp.x_lb + i*L->St_sp.d_x;
        }
        for (int j = 0; j<L->St_sp.m_2; j++) {
            y_r[j]    = gsl_vector_get(&center_row.vector, j);
            axis_y[j] = L->St_sp.y_lb + j*L->St_sp.d_y;
        }
        gsl_interp_accel * acc_x = gsl_interp_accel_alloc();
        gsl_interp_accel * acc_y = gsl_interp_accel_alloc();
        gsl_spline * spline_x    = gsl_spline_alloc(gsl_interp_cspline, m_1);
        gsl_spline * spline_y    = gsl_spline_alloc(gsl_interp_cspline, m_2);
        gsl_spline_init(spline_x, axis_x, x_c, L->St_sp.m_1);
        gsl_spline_init(spline_y, axis_y, y_r, L->St_sp.m_2);
        Out.delta_1_interpol = gsl_spline_eval_deriv(spline_x, L->St_sp.x_lb + c_1*L->St_sp.d_x, acc_x);
        Out.delta_2_interpol = gsl_spline_eval_deriv(spline_y, L->St_sp.y_lb + c_2*L->St_sp.d_y, acc_y);
        Out.gamma_1_interpol = gsl_spline_eval_deriv2(spline_x, L->St_sp.x_lb + c_1*L->St_sp.d_x, acc_x);
        Out.gamma_2_interpol = gsl_spline_eval_deriv2(spline_y, L->St_sp.y_lb + c_2*L->St_sp.d_y, acc_y);
        gsl_spline_free(spline_x);
        gsl_spline_free(spline_y);
        gsl_interp_accel_free(acc_x);
        gsl_interp_accel_free(acc_y);
        return Out;
    }
};


#endif /* adi_h */