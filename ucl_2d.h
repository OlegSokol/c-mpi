//  ucl_2d.h
//  PDE_solver
//  Created by OLEG SOKOLINSKIY on 9/17/16.
//  Copyright © 2016 OLEG SOKOLINSKIY. All rights reserved.


/*
DESCRIPTION: CREATES L (L_1 and L_2) OPERATORS FOR ADI NUMERICAL SOLUTION OF A BIVARIATE PDE
*/

#ifndef ucl_2d_h
#define ucl_2d_h

#include <vector>

#include "models.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>


//description of the state-space
struct st_sp {
    int    m_1;
    double x_lb, d_x;
    
    int    m_2;
    double y_lb, d_y;
};

//lower center upper diagonals for tri-diagonal systems
struct lcu {
    gsl_vector * l;
    gsl_vector * c;
    gsl_vector * u;
    
    void free_lcu() //destructor must free gsl vectors
    {
        gsl_vector_free(l);
        gsl_vector_free(c);
        gsl_vector_free(u);
    }
};



class L_operator {
    
public:
    
    L_operator(st_sp St_sp_, model_2d * Model_) //constructs L operator for a 2-dim model (PDE) given a state-space specification
    :St_sp(St_sp_), Model(Model_), STATUS(-100)
    {
        DIAG_L_1_m1.resize(St_sp.m_1);
        DIAG_L_1_m2.resize(St_sp.m_2);
        DIAG_L_2_m1.resize(St_sp.m_1);
        DIAG_L_2_m2.resize(St_sp.m_2);
        for (int i=0; i<St_sp.m_2; i++) {              // main work is done by the private function calc_diagonals() - see below
            calc_diagonals(1, 1, i, &DIAG_L_1_m2[i]);  // L_1 - ADI direction 1 --> for different fixed j_{2} - length of main diagonal m_1                            [LHS]
            calc_diagonals(2, 1, i, &DIAG_L_2_m2[i]);  // L_2 - ADI direction 1 --> for different fixed j_{2} - length of ALL diagonals m_1 and no boundary conditions [RHS]
        }
        for (int i=0; i<St_sp.m_1; i++) {
            calc_diagonals(2, 2, i, &DIAG_L_2_m1[i]);  // L_2 - ADI direction 2 for different fixed j_{1} - length of main diagonal m_2                            [LHS]
            calc_diagonals(1, 2, i, &DIAG_L_1_m1[i]);  // L_1 - ADI direction 2 for different fixed j_{1} - length of ALL diagonals m_2 and no boundary conditions [RHS]
        }
    }
    
    //destructors: must free memory
    ~L_operator() {
        if (STATUS != -100) {
            for (std::vector<lcu>::iterator it = DIAG_L_1_m1.begin(); it != DIAG_L_1_m1.end(); ++it) {
                it->free_lcu();
            }
            for (std::vector<lcu>::iterator it = DIAG_L_1_m2.begin(); it != DIAG_L_1_m2.end(); ++it) {
                it->free_lcu();
            }
            for (std::vector<lcu>::iterator it = DIAG_L_2_m1.begin(); it != DIAG_L_2_m1.end(); ++it) {
                it->free_lcu();
            }
            for (std::vector<lcu>::iterator it = DIAG_L_2_m2.begin(); it != DIAG_L_2_m2.end(); ++it) {
                it->free_lcu();
            }
        }
    }

    std::vector<lcu> DIAG_L_1_m1, DIAG_L_1_m2;
    std::vector<lcu> DIAG_L_2_m1, DIAG_L_2_m2;
    st_sp      St_sp;
    model_2d * Model;


private:
    int STATUS;
    typedef double (model_2d::* coef)(double, double);
    void calc_diagonals(int L_operator, int direction, int state_index, lcu * output) //main work for creating the diagonals of the tri-diagonal matrices L_1 and L_2 (int L_operator); output depends on ADI direction (int direction)
    {
        int m;
        double delta, delta_sq;
        double x, y;
        coef drift_func;
        coef diffu_func;
        double Diffusion, Drift, Rate;
        //**************************************************   L OPERATORS -> CHOICE **************************************************//
        if (L_operator == 1) {                        // L_1 operator
            delta      = St_sp.d_x;
            delta_sq   = gsl_pow_2(St_sp.d_x);
            drift_func = &model_2d::mu_1;
            diffu_func = &model_2d::sigma_1;
        }
        else {
            delta    = St_sp.d_y;                     // L_2 operator
            delta_sq = gsl_pow_2(St_sp.d_y);
            drift_func = &model_2d::mu_2;
            diffu_func = &model_2d::sigma_2;
        }
        //**************************************************   ADI DIRECTION 1 **************************************************//
        if(direction == 1) {
            m         = St_sp.m_1;                    //This sets the length of the diagonals l,c,u; ADI direction 1 solves systems of m_1 equations
            
            y = St_sp.y_lb + St_sp.d_y*state_index;   //ADI direction 1 solves m_2 systems of m_1 equations for varying fixed j_{2}'s (j_{2} in 1:m_2)
            
            
            //**************************************************   BLOCK LHS of ADI equations **************************************************//
            if (direction == L_operator) {
                
                output->u = gsl_vector_calloc(m-1);   //AP:  u_{1} u_{2} ... u_{m-1}        -- 0 ... m-1  - u[i] evaluated at x[i], y[i]     -- NOTE: NO u_{m-1}
                output->c = gsl_vector_calloc(m);     //AP:  c_{1} c_{2} ... c_{m-1} c_{m}  -- 0 ... m    - c[i] evaluated at x[i], y[i]
                output->l = gsl_vector_calloc(m-1);   //AP:        l_{2} ...         l_{m}  -- 0 ... m-1  - l[i] evaluated at x[i+1], y[i+1] -- NOTE: NO l_{1}
                
                x         = St_sp.x_lb;
                Rate      = Model->rf_rate(x,y)/2;
                Drift     = (Model->*drift_func)(x,y)/delta;
                Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                gsl_vector_set(output->c, 0, -Diffusion - Rate      + 2 * (-Drift/2 + Diffusion/2) ); // l[0] = AP_ind '2'  c[0] = AP_ind '1'         AP: c_{1} = c_{1} + 2 * l_{1}
                gsl_vector_set(output->u, 0,  Drift/2 + Diffusion/2 -     (-Drift/2 + Diffusion/2) ); // u[0] = AP_ind '1'  c[0] = AP_ind '1'         AP: u_{1} = u_{1} -     l_{1}
                
                for (int i = 1; i<m-1; i++) {
                    x         = St_sp.x_lb + St_sp.d_x*i;
                    Drift     = (Model->*drift_func)(x,y)/delta;
                    Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                    Rate      = Model->rf_rate(x,y)/2;
                    gsl_vector_set(output->l, i-1, -Drift/2 + Diffusion/2);                           // l[i-1] evaluated at x[i], y[i]
                    gsl_vector_set(output->c, i,   -Diffusion - Rate);
                    gsl_vector_set(output->u, i,    Drift/2 + Diffusion/2);
                }
                
                x         = St_sp.x_lb + St_sp.d_x*(m-1);
                Drift     = (Model->*drift_func)(x,y)/delta;
                Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                Rate      = Model->rf_rate(x,y)/2;
                gsl_vector_set(output->c, m-1, -Diffusion - Rate);
                gsl_vector_set(output->l, m-2, -Drift/2 + Diffusion/2);
                gsl_vector_set(output->c, m-1, gsl_vector_get(output->c, m-1) + 2 * (Drift/2 + Diffusion/2) );  // u[m-2] = AP_ind 'm-1'  c[m-1] = AP_ind 'm'   AP: c_{m} = c_{m} + 2 * u_{m}
                gsl_vector_set(output->l, m-2, gsl_vector_get(output->l, m-2) -     (Drift/2 + Diffusion/2) );  // l[m-2] = AP_ind 'm'    c[m-1] = AP_ind 'm'   AP: l_{m} = l_{m} -     u_{m}
                
            }
            //**************************************************   BLOCK RHS of ADI equations **************************************************//
            else {
                output->l = gsl_vector_calloc(m);   //AP:  l_{1} l_{2} ... l_{m}  -- 0 ... m-1  - evaluated at x[i], y[i]  -- NOTE: presence of l_{1}
                output->c = gsl_vector_calloc(m);   //AP:  c_{1} c_{2} ... c_{m}  -- 0 ... m-1  - evaluated at x[i], y[i]
                output->u = gsl_vector_calloc(m);   //AP:  u_{1} u_{2} ... u_{m}  -- 0 ... m-1  - evaluated at x[i], y[i]  -- NOTE: presence of u_{m}
                
                                                                                    // RHS of ADI: NO boundary conditions
                if (state_index == 0) {
                    for (int i = 0; i<m; i++) {
                        x         = St_sp.x_lb + St_sp.d_x*i;
                        Drift     = (Model->*drift_func)(x,y)/delta;
                        Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                        Rate      = Model->rf_rate(x,y)/2;
                        gsl_vector_set(output->l, i, 0);                              // RHS of ADI: l[i] evaluated at x[i], y[i]
                        gsl_vector_set(output->c, i, -Drift - Rate);
                        gsl_vector_set(output->u, i,  Drift);
                    }
                }
                else if (state_index == St_sp.m_2-1) {
                    for (int i = 0; i<m; i++) {
                        x         = St_sp.x_lb + St_sp.d_x*i;
                        Drift     = (Model->*drift_func)(x,y)/delta;
                        Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                        Rate      = Model->rf_rate(x,y)/2;
                        gsl_vector_set(output->u, i, 0);                              // RHS of ADI: l[i] evaluated at x[i], y[i]
                        gsl_vector_set(output->c, i, Drift - Rate);
                        gsl_vector_set(output->l, i, -Drift);
                    }
                }
                else {
                    for (int i = 0; i<m; i++) {
                        x         = St_sp.x_lb + St_sp.d_x*i;
                        Drift     = (Model->*drift_func)(x,y)/delta;
                        Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                        Rate      = Model->rf_rate(x,y)/2;
                        gsl_vector_set(output->l, i, -Drift/2 + Diffusion/2);          // RHS of ADI: l[i] evaluated at x[i], y[i]
                        gsl_vector_set(output->c, i, -Diffusion - Rate);
                        gsl_vector_set(output->u, i,  Drift/2 + Diffusion/2);
                    }
                }
            }
        }


        //**************************************************   ADI DIRECTION 2 **************************************************//
        else {
            m     = St_sp.m_2;
            x = St_sp.x_lb + St_sp.d_x*state_index;
            //**************************************************   BLOCK LHS of ADI equations **************************************************//
            if (direction == L_operator) {
                output->l = gsl_vector_calloc(m-1);
                output->c = gsl_vector_calloc(m);
                output->u = gsl_vector_calloc(m-1);
                y         = St_sp.y_lb;
                Drift     = (Model->*drift_func)(x,y)/delta;
                Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                Rate      = Model->rf_rate(x,y)/2;
                gsl_vector_set(output->c, 0, -Diffusion - Rate      + 2 * (-Drift/2 + Diffusion/2) ); // l[0] = AP_ind '2'  c[0] = AP_ind '1'         AP: c_{1} = c_{1} + 2 * l_{1}
                gsl_vector_set(output->u, 0,  Drift/2 + Diffusion/2 -     (-Drift/2 + Diffusion/2) ); // u[0] = AP_ind '1'  c[0] = AP_ind '1'         AP: u_{1} = u_{1} -     l_{1}
                for (int i = 1; i<m-1; i++) {
                    y         = St_sp.y_lb + St_sp.d_y*i;
                    Drift     = (Model->*drift_func)(x,y)/delta;
                    Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                    Rate      = Model->rf_rate(x,y)/2;
                    gsl_vector_set(output->l, i-1, -Drift/2 + Diffusion/2);
                    gsl_vector_set(output->c, i,   -Diffusion - Rate);
                    gsl_vector_set(output->u, i,    Drift/2 + Diffusion/2);
                }
                y         = St_sp.y_lb + St_sp.d_y*(m-1);
                Drift     = (Model->*drift_func)(x,y)/delta;
                Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                Rate      = Model->rf_rate(x,y)/2;
                gsl_vector_set(output->c, m-1, -Diffusion - Rate);
                gsl_vector_set(output->l, m-2, -Drift/2 + Diffusion/2);
                gsl_vector_set(output->c, m-1, gsl_vector_get(output->c, m-1) + 2 * (Drift/2 + Diffusion/2) );  // u[m-2] = AP_ind 'm-1'  c[m-1] = AP_ind 'm'   AP: c_{m} = c_{m} + 2 * u_{m}
                gsl_vector_set(output->l, m-2, gsl_vector_get(output->l, m-2) -     (Drift/2 + Diffusion/2) );  // l[m-2] = AP_ind 'm'    c[m-1] = AP_ind 'm'   AP: l_{m} = l_{m} -     u_{m}
            }
            //**************************************************   BLOCK RHS of ADI equations **************************************************//
           else {
                output->l = gsl_vector_calloc(m);
                output->c = gsl_vector_calloc(m);
                output->u = gsl_vector_calloc(m);
               
               if (state_index == 0) {
                   for (int i = 0; i<m; i++) {
                       y         = St_sp.y_lb + St_sp.d_y*i;
                       Drift     = (Model->*drift_func)(x,y)/delta;
                       Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                       Rate      = Model->rf_rate(x,y)/2;
                       gsl_vector_set(output->l, i, 0);                              // RHS of ADI: l[i] evaluated at x[i], y[i]
                       gsl_vector_set(output->c, i, -Drift - Rate);
                       gsl_vector_set(output->u, i,  Drift);
                   }
               }
               else if (state_index == St_sp.m_1-1) {
                   for (int i = 0; i<m; i++) {
                       y         = St_sp.y_lb + St_sp.d_y*i;
                       Drift     = (Model->*drift_func)(x,y)/delta;
                       Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                       Rate      = Model->rf_rate(x,y)/2;
                       gsl_vector_set(output->u, i, 0);                              // RHS of ADI: l[i] evaluated at x[i], y[i]
                       gsl_vector_set(output->c, i, Drift - Rate);
                       gsl_vector_set(output->l, i, -Drift);
                   }
               }
               else {
                   for (int i = 0; i<m; i++) {
                       y         = St_sp.y_lb + St_sp.d_y*i;
                       Drift     = (Model->*drift_func)(x,y)/delta;
                       Diffusion = gsl_pow_2((Model->*diffu_func)(x,y))/delta_sq;
                       Rate      = Model->rf_rate(x,y)/2;
                       gsl_vector_set(output->l, i, -Drift/2 + Diffusion/2);
                       gsl_vector_set(output->c, i, -Diffusion - Rate);
                       gsl_vector_set(output->u, i,  Drift/2 + Diffusion/2);
                   }
               }
            }
        }
        STATUS = 0;
    }
};


#endif /* ucl_2d_h */