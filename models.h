//  model.h
//  PDE_solver
//  Created by OLEG SOKOLINSKIY on 9/17/16.
//  Copyright © 2016 OLEG SOKOLINSKIY. All rights reserved.


#ifndef model_h
#define model_h

#include <math.h>
#include <gsl/gsl_math.h>

#include <iostream>

//pure virtual base class for 2-dimensional models
class model_2d {
public:
    virtual double rf_rate(double x, double y) = 0; //drift parameters
    virtual double mu_1(double x, double y) = 0;
    virtual double mu_2(double x, double y) = 0;
    virtual double sigma_1(double x, double y) = 0; //diffusion parameters
    virtual double sigma_2(double x, double y) = 0;
    virtual double yield(double R, double Maturity) = 0; //analytic yield calculation if the model is for a bond AND the bond is option-free
    virtual double ZCB(double R, double Maturity)   = 0; //analytic zero coupon bond price if the model is for a bond AND the bond is option-free
};


//stochastic volatility model - used for testing ADI with equity derivatives -> yield() and ZCB() functions are placeholders
class model_SV : public model_2d {
private:
    double rate, cs, lrv, vv, div_y; //cs - mean-reversion speed; lrv - LR vol level; vv - vol of variance
    
public:
    model_SV(double R, double CS, double LRV, double VV, double Div_y = 0)
    : rate(R), cs(CS), lrv(LRV), vv(VV), div_y(Div_y)
    {}
    
    double rf_rate(double x, double y) {
        return rate;
    }
    
    double mu_1(double x, double y) {
        return (rate-div_y)*x;
    }
    
    double mu_2(double x, double y) {
        return cs*(lrv - y);
    }
    
    double sigma_1(double x, double y) {
        return x*sqrt(y);
    }
    
    double sigma_2(double x, double y) {
        return vv*sqrt(y);
    }
    
    double yield(double R, double Maturity) {
        return 0.0;
    }
    
    double ZCB(double R, double Maturity) {
        return 0.0;
    }
    
};



//structural model for corporate bond pricing:
// V - firm asset value - credit event - firm asset value dropping below a barrier <- accounts for CREDIT risk
// R - short rate - CIR dynamics <- account for INTEREST RATE risk
class model_VR_Structural : public model_2d {

private:
    double vol_V, cs_R, lr_R, vol_R, div_y;

public:
    
    model_VR_Structural(double Vol_V, double Cs_R, double Lr_R, double Vol_R, double Div_y = 0)
    : vol_V(Vol_V), cs_R(Cs_R), lr_R(Lr_R), vol_R(Vol_R), div_y(Div_y)
    {}
    
    double rf_rate(double x, double y) {
        return y;
    }
    
    double mu_1(double x, double y) {
        return 0;//(y-div_y)*x;
    }
    
    double mu_2(double x, double y) {
        return cs_R*(lr_R - y);
    }
    
    double sigma_1(double x, double y) {
        return vol_V*x;
    }
    
    double sigma_2(double x, double y) {
        return vol_R*sqrt(y);
    }
    
    //analytic yield calculation if the model is for a bond AND the bond is option-free
    double yield(double R, double Maturity) {
        if (Maturity==0) return R;
        double h = sqrt( gsl_pow_2(cs_R) + 2*gsl_pow_2(vol_R) );
        double p = 2*cs_R*lr_R/gsl_pow_2(vol_R);
        double A = pow( 2*h*exp((cs_R+h)*Maturity/2) / ( 2*h + (cs_R+h)*(exp(Maturity*h)-1) ), p );
        double B = 2*(exp(Maturity*h)-1) / ( 2*h + (cs_R+h)*(exp(Maturity*h)-1) );
        return -(log(A) - B*R)/Maturity;
    }
    
    //analytic zero coupon bond price if the model is for a bond AND the bond is option-free
    double ZCB(double R, double Maturity) {
        double h = sqrt( gsl_pow_2(cs_R) + 2*gsl_pow_2(vol_R) );
        double p = 2*cs_R*lr_R/gsl_pow_2(vol_R);
        double A = pow( 2*h*exp((cs_R+h)*Maturity/2) / ( 2*h + (cs_R+h)*(exp(Maturity*h)-1) ), p );
        double B = 2*(exp(Maturity*h)-1) / ( 2*h + (cs_R+h)*(exp(Maturity*h)-1) );
        return A*exp(-B*R);
    }
    
    
};



//one dimensional model
class model_1d {
public:
    virtual double rf_rate(void) = 0;
    virtual double mu(double x) = 0;
    virtual double sigma(double x) = 0;

};


//Black-Scholes model - Geometric Brownian motion
class model_BS : public model_1d {
private:
    double rate, Sigma;
public:
    model_BS(double R, double Sigma_)
    : rate(R), Sigma(Sigma_)
    {}
    
    double rf_rate(void) {
        return rate;
    }
    
    double mu(double x) {
        return rate*x;
    }

    double sigma(double x) {
        return Sigma*x;
    }
};

#endif /* model_h */