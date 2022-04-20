#ifndef CERES_COSTFUN_H
#define CERES_COSTFUN_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>


using namespace ceres;


struct HomographyCost2
{
    HomographyCost2(double xw, double yw, double zw, double u, double v):xw(xw),yw(yw),zw(zw),u(u),v(v) {};

    template<typename T>
    bool operator()(const T* const h, T* residual) const
    {
        T u_ = T(h[0] * xw + h[1] * yw + h[2]*zw+h[3]);
        T v_ = T(h[4] * xw + h[5] * yw + h[6]*zw+h[7]);

        residual[0] = ceres::sqrt(ceres::pow(T(u_) - u, 2) + ceres::pow(T(v_) - v, 2));
        return true;
    }

    const double xw, yw, zw, u, v;
    

};
struct HomographyCost {

    HomographyCost(double x1, double y1,double x2, double y2)
        :x1(x1), y1(y1), x2(x2), y2(y2)
    {}

    template<typename T>
    bool operator()(const T *const h, T* residual) const
    {
        T w = T(h[2]*x1+h[5]*y1+h[8]);
        T x = T(h[0]*x1+h[3]*y1+h[6])/w;
        T y = T(h[1]*x1+h[4]*y1+h[7])/w;

        residual[0] = ceres::sqrt(ceres::pow(T(x2)-x, 2) + ceres::pow(T(y2)-y, 2));
        return true;
    }
    const double x1, x2, y1, y2;
};


struct ProjectCost {

    Eigen::Vector3d objPt;
    Eigen::Vector2d imgPt;
    const int mDistortionParaNum;

    ProjectCost(Eigen::Vector3d& objPt, Eigen::Vector2d& imgPt, const int& paraNum):
        objPt(objPt), imgPt(imgPt),mDistortionParaNum(paraNum){}

    template<typename T>
    bool operator()(
        const T *const k,
        const T *const r,
        const T *const t,
        T* residuals)const
    {
        T pos3d[3] = {T(objPt(0)), T(objPt(1)), T(objPt(2))};
        T pos3d_proj[3];
        // 旋转
        ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
        // 平移
        pos3d_proj[0] += t[0];
        pos3d_proj[1] += t[1];
        pos3d_proj[2] += t[2];

        T xp = pos3d_proj[0] / pos3d_proj[2];
        T yp = pos3d_proj[1] / pos3d_proj[2];

        T xdis = T(0.0);
        T ydis = T(0.0);

        const T& fx = k[0];
        const T& fy = k[1];
        const T& cx = k[2];
        const T& cy = k[3];

        if (mDistortionParaNum == 5)
        {
            const T& k1 = k[4];
            const T& k2 = k[5];
            const T& k3 = k[6];

            const T& p1 = k[7];
            const T& p2 = k[8];

            // 径向畸变
            T r_2 = xp * xp + yp * yp;

            xdis = xp * (T(1.) + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + T(2.) * p1 * xp * yp + p2 * (r_2 + T(2.) * xp * xp);
            ydis = yp * (T(1.) + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + p1 * (r_2 + T(2.) * yp * yp) + T(2.) * p2 * xp * yp;
        }
        else if (mDistortionParaNum == 4)
        {
            const T& k1 = k[4];
            const T& k2 = k[5];

            const T& p1 = k[6];
            const T& p2 = k[7];

            // 径向畸变
            T r_2 = xp * xp + yp * yp;

            xdis = xp * (T(1.) + k1 * r_2 + k2 * r_2 * r_2) + T(2.) * p1 * xp * yp + p2 * (r_2 + T(2.) * xp * xp);
            ydis = yp * (T(1.) + k1 * r_2 + k2 * r_2 * r_2) + p1 * (r_2 + T(2.) * yp * yp) + T(2.) * p2 * xp * yp;
        }
       
        
        // 像素距离
        T u = fx*xdis + cx;
        T v = fy*ydis + cy;

        residuals[0] = u - T(imgPt[0]);
        residuals[1] = v - T(imgPt[1]);
        return true;
    }
};


struct ProjectCostRT {

    Eigen::Vector3d objPt;
    Eigen::Vector2d imgPt;
    Eigen::Matrix3d mK;

    ProjectCostRT(const Eigen::Vector3d& objPt, const Eigen::Vector2d& imgPt, const Eigen::Matrix3d& K) :
        objPt(objPt), imgPt(imgPt), mK(K) {}

    template<typename T>
    bool operator()(
        const T* const r,
        const T* const t,
        T* residuals)const
    {
        T pos3d[3] = { T(objPt(0)), T(objPt(1)), T(objPt(2)) };
        T pos3d_proj[3];
        // 旋转
        ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
        // 平移
        pos3d_proj[0] += t[0];
        pos3d_proj[1] += t[1];
        pos3d_proj[2] += t[2];

        T xp = pos3d_proj[0] / pos3d_proj[2];
        T yp = pos3d_proj[1] / pos3d_proj[2];

        const T& fx = T(mK(0,0));
        const T& fy = T(mK(1,1));
        const T& cx = T(mK(0,2));
        const T& cy = T(mK(1,2));

        // 像素距离
        T u = fx * xp + cx;
        T v = fy * yp + cy;

        residuals[0] = u - T(imgPt[0]);
        residuals[1] = v - T(imgPt[1]);
        return true;
    }
};

struct BundleAdjustment
{
    double imgx;
    double imgy;

    Eigen::Matrix3d mK;
    
    BundleAdjustment(const double& u, const double& v, const Eigen::Matrix3d& K) :imgx(u), imgy(v), mK(K) {};
    
    template<typename T>
    bool operator()(
        const T* const p,
        const T* const r,
        const T* const t,
        T* residuals)const
    {
        const T fx = T(mK(0, 0));
        const T fy = T(mK(1, 1));
        const T cx = T(mK(0, 2));
        const T cy = T(mK(1, 2));

        T pos3d_proj[3];
        ceres::AngleAxisRotatePoint(r, p, pos3d_proj);
        
        pos3d_proj[0] += t[0];
        pos3d_proj[1] += t[1];
        pos3d_proj[2] += t[2];
        
        T xp = pos3d_proj[0] / pos3d_proj[2];
        T yp = pos3d_proj[1] / pos3d_proj[2];

        T u_pre = fx * xp + cx;
        T v_pre = fy * yp + cy;
        
        residuals[0] = u_pre - imgx;
        residuals[1] = v_pre - imgy;

        return true;
    }
};
#endif
