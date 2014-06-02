/********************************************************************

   fdtd.cpp -

   $Author: Wang Wen $
   created at: Sun Dec 15 17:42:34 JST 2013

   Copyright (C) 2013-2014 Wang Wen

*********************************************************************/


#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

#include <blitz/array.h>
#include <Python.h>
#include <omp.h>

#define FDTD_PYTHON_MODULE
#define private public

#ifdef _WIN32

inline double round( double d )
{
    return floor(d + 0.5);
}

#endif


namespace fdtd {

#ifdef __GNUG__
const double pi = M_PI;

#elif _WIN32
const double pi = 3.141592653589793238463;

#endif

#ifdef __GNUG__
const double inf = 1.0 / 0.0;
#elif _WIN32
const double inf = 1e100;
#endif

const double eps_0 = 8.854187817e-12;
const double mu_0 = 4e-7 * pi;
const double c = 2.998e8;

class vector_3 {

public:

    double x;
    double y;
    double z;

    vector_3(double x, double y, double z): x(x), y(y), z(z) {

    }

#ifdef FDTD_PYTHON_MODULE

    vector_3(PyObject *p) {
        if(PySequence_Check(p)) {
            x = PyFloat_AsDouble(PySequence_GetItem(p, 0));
            y = PyFloat_AsDouble(PySequence_GetItem(p, 1));
            z = PyFloat_AsDouble(PySequence_GetItem(p, 2));
        }
        else {
            //std::cout << "Error vector initialized from python object, the argument is not a sequence...";
            x = PyFloat_AsDouble(PyObject_GetAttrString(p, "x"));
            y = PyFloat_AsDouble(PyObject_GetAttrString(p, "y"));
            z = PyFloat_AsDouble(PyObject_GetAttrString(p, "z"));
        }

    }

    PyObject *tuple() const {
        PyObject *tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(x));
        PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(y));
        PyTuple_SetItem(tuple, 2, PyFloat_FromDouble(z));
        return tuple;
    }

#endif

    vector_3(): x(0), y(0), z(0) {

    }

    std::string str() const;

private:


};

std::ostream& operator<<(std::ostream& os, const vector_3 &pt) {
    os << "(" << pt.x;
    os << ", " << pt.y;
    os << ", " << pt.z;
    os << ")";
    return os;
}

std::string vector_3::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

inline vector_3 operator+(const vector_3 &vct1, const vector_3 &vct2) {
    return vector_3(vct1.x + vct2.x,
            vct1.y + vct2.y,
            vct1.z + vct2.z
        );
}

inline vector_3 operator+(const vector_3 &vct, double val) {
    return vector_3(vct.x + val,
            vct.y + val,
            vct.z + val
        );
}

inline vector_3 operator+(const vector_3 &vct1, PyObject *vct2) {
    return vector_3(vct1.x + PyFloat_AsDouble(PySequence_GetItem(vct2, 0)),
            vct1.y + PyFloat_AsDouble(PySequence_GetItem(vct2, 1)),
            vct1.z + PyFloat_AsDouble(PySequence_GetItem(vct2, 2))
        );
}

inline vector_3 operator+(PyObject *vct1, const vector_3 &vct2) {
    return vector_3(PyFloat_AsDouble(PySequence_GetItem(vct1, 0)) + vct2.x,
            PyFloat_AsDouble(PySequence_GetItem(vct1, 1)) + vct2.y,
            PyFloat_AsDouble(PySequence_GetItem(vct1, 2)) + vct2.z
        );
}

inline vector_3 operator-(const vector_3 &vct1, const vector_3 &vct2) {
    return vector_3(vct1.x - vct2.x,
            vct1.y - vct2.y,
            vct1.z - vct2.z
        );
}

inline vector_3 operator-(const vector_3 &vct1, PyObject *vct2) {
    return vector_3(vct1.x - PyFloat_AsDouble(PySequence_GetItem(vct2, 0)),
            vct1.y - PyFloat_AsDouble(PySequence_GetItem(vct2, 1)),
            vct1.z - PyFloat_AsDouble(PySequence_GetItem(vct2, 2))
        );
}

inline vector_3 operator-(PyObject *vct1, const vector_3 &vct2) {
    return vector_3(PyFloat_AsDouble(PySequence_GetItem(vct1, 0)) - vct2.x,
            PyFloat_AsDouble(PySequence_GetItem(vct1, 1)) - vct2.y,
            PyFloat_AsDouble(PySequence_GetItem(vct1, 2)) - vct2.z
        );
}

inline vector_3 operator-(const vector_3 &vct, double val) {
    return vector_3(vct.x - val,
            vct.y - val,
            vct.z - val
        );
}

inline vector_3 operator*(double k, const vector_3 &vct) {
    return vector_3(k * vct.x,
            k * vct.y,
            k * vct.z
        );
}

inline vector_3 operator*(const vector_3 &vct, double k) {
    return vector_3(k * vct.x,
            k * vct.y,
            k * vct.z
        );
}

inline vector_3 operator/(const vector_3 &vct, double k) {
    return vector_3(vct.x / k,
            vct.y / k,
            vct.z / k
        );
}

inline double norm(const vector_3 &vct) {
    return sqrt(vct.x * vct.x +
            vct.y * vct.y +
            vct.z * vct.z);
}

inline double inner_product(const vector_3 &vct1, const vector_3 &vct2) {
    return (vct1.x * vct2.x +
            vct1.y * vct2.y +
            vct1.z * vct2.z);
}

inline vector_3 cross_product(const vector_3 &vct1, const vector_3 &vct2) {
    return vector_3(vct1.y * vct2.z - vct1.z * vct2.y,
            vct1.z * vct2.x - vct1.x * vct2.z,
            vct1.x * vct2.y - vct1.y * vct2.x);
}

typedef vector_3 point_3;
typedef vector_3 coord_3;

typedef blitz::TinyVector<int, 3> Index3;

const vector_3 VEC_ORIGIN = vector_3(0, 0, 0);
const vector_3 VEC_UNIT_X = vector_3(1, 0, 0);
const vector_3 VEC_UNIT_Y = vector_3(0, 1, 0);
const vector_3 VEC_UNIT_Z = vector_3(0, 0, 1);

void Assert(bool statement, std::string err) {
    if(!statement) {
        std::cout << err << std::endl;
        exit(1);
    }
}

bool IntValue(double f) {
    return ((double)((int)(f)) == f);
}

bool ProperDivide(double a, double b, double h) {
    return (a > b) && (IntValue((b - a) / h));
}

double GaussianPulse(double t, double tao) {
    return exp(-pow(t / tao, 2));
}

double SinWave(double t, double omega = 2 * pi) {
    return sin(omega * t);
}

double dd_gaussian(double t, double k) {
    return pow(32.0 * pow(k, 2) / (9 * pi), 0.25)
            * (1.0 - 2.0 * pow(k * t, 2))
            * exp(- pow(k * t, 2));
}

double Min(double a, double b) {
    return ((a < b) ? a: b);
}

int Min(int a, int b) {
    return ((a < b) ? a: b);
}

double Max(double a, double b) {
    return ((a > b) ? a: b);
}

int Max(int a, int b) {
    return ((a > b) ? a: b);
}

struct Medium {

    Medium(): eps(eps_0), mu(mu_0), sig(0) {

    }

    Medium(double eps, double mu, double sig):
        eps(eps), mu(mu), sig(sig) {

    }

    double eps;
    double mu;
    double sig;
};


class Shape {

public:

    virtual bool hasInnerPoint(point_3 pt) const = 0;

    virtual double max_x() const = 0;
    virtual double max_y() const = 0;
    virtual double max_z() const = 0;

    virtual double min_x() const = 0;
    virtual double min_y() const = 0;
    virtual double min_z() const = 0;

    virtual ~Shape() {

    }

};



class Brick: public Shape {

public:

    Brick() {

    }

    Brick(const vector_3 &min, const vector_3 &max):
        coord_min(min), coord_max(max) {

    }

#ifdef FDTD_PYTHON_MODULE

    Brick(PyObject *min, PyObject *max):
        coord_min(min), coord_max(max) {

    }

#endif

    std::string str() const;

    bool hasInnerPoint(point_3 pt) const override {
        return (pt.x < coord_max.x && pt.x > coord_min.x &&
                pt.y < coord_max.y && pt.y > coord_min.y &&
                pt.z < coord_max.z && pt.z > coord_min.z );
    }

    double max_x() const override {
        return coord_max.x;
    }

    double max_y() const override {
        return coord_max.y;
    }

    double max_z() const override {
        return coord_max.z;
    }

    double min_x() const override {
        return coord_min.x;
    }

    double min_y() const override {
        return coord_min.y;
    }

    double min_z() const override {
        return coord_min.z;
    }


private:

    coord_3 coord_max;
    coord_3 coord_min;

};

std::ostream& operator<<(std::ostream& os, Brick cub) {
    os << "geometrical object cuboid with bounded field:" << std::endl;
    os << "   X axis in range (" << cub.min_x() << ", " << cub.max_x() << ")" << std::endl;
    os << "   Y axis in range (" << cub.min_y() << ", " << cub.max_y() << ")" << std::endl;
    os << "   Z axis in range (" << cub.min_z() << ", " << cub.max_z() << ")" << std::endl;
    return os;
}

std::string Brick::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}



class Sphere: public Shape {

public:

    Sphere(): radius(1), center() {

    }

    Sphere(const point_3 &cnt, double radi): radius(radi), center(cnt) {

    }

#ifdef FDTD_PYTHON_MODULE

    Sphere(PyObject *cnt, double radi): radius(radi), center(cnt) {

    }

#endif

    bool hasInnerPoint(point_3 pt) const override {
        return norm(pt - center) < radius;
    }

    point_3 getCenter() const {
        return center;
    }

    double getRadius() {
        return radius;
    }

    double max_x() const override {
        return center.x + radius;
    }

    double max_y() const override {
        return center.y + radius;
    }

    double max_z() const override {
        return center.z + radius;
    }

    double min_x() const override {
        return center.x - radius;
    }

    double min_y() const override {
        return center.y - radius;
    }

    double min_z() const override {
        return center.z - radius;
    }

    std::string str() const;

private:

    double radius;
    point_3 center;

};

std::ostream& operator<<(std::ostream& os, Sphere sph) {
    std::cout << "geometrical object sphere, with center at " << sph.getCenter()
            << " and radius of " << sph.getRadius() << "..."
            << std::endl;
    return os;
}

std::string Sphere::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

class Cartesian {

public:

    Cartesian():coord_max(), coord_min(), resol(0) {

    }
    Cartesian(const coord_3 &max, const coord_3 &min, double resol);
    Cartesian(const coord_3 &max, double resol);

    Cartesian(PyObject *max, double resol) {
        vector_3 max_v(max);
        this->resol = resol;
        this->coord_max = this->fit_resol(max);
        this->coord_min = coord_3(0, 0, 0);
    }

    point_3 point_at(double x, double y, double z) const;

    std::string str() const;
/**
    Cartesian &addBrickMedium(const Brick &shp, const Medium &med) {
        this->dlcf.insert(std::pair<boost::any, Medium>(shp, med));
        return *this;
    }

    Cartesian &addSphereMedium(const Sphere &shp, const Medium &med) {
        this->dlcf.insert(std::pair<boost::any, Medium>(shp, med));
        return *this;
    }
**/
    double resolu() const {
        return resol;
    }

    double max_x() const
    {
        return coord_max.x;
    }

    double max_y() const
    {
        return coord_max.y;
    }

    double max_z() const
    {
        return coord_max.z;
    }

    double min_x() const
    {
        return coord_min.x;
    }

    double min_y() const
    {
        return coord_min.y;
    }

    double min_z() const
    {
        return coord_min.z;
    }

private:


    coord_3 coord_max;
    coord_3 coord_min;

    double resol;

    //std::map<boost::any, Medium> dlcf;
    //std::vector<Shape> shl;

    double fit_resol(double length) const;
    coord_3 fit_resol(const coord_3 &coord) const;

};

double Cartesian::fit_resol(double length) const {
    double base = (int)(length / resol) * resol;
    double remain = length - base;

    // std::cout << fabs(remain) / resol << std::endl;

    if(fabs(remain) / resol > 0.5) {
        if(remain > 0)
            return base + resol;
        return base - resol;
    }
    return base;
}

coord_3 Cartesian::fit_resol(const coord_3 &coord) const {
    return coord_3(this->fit_resol(coord.x),
            this->fit_resol(coord.y),
            this->fit_resol(coord.z)
        );
}

Cartesian::Cartesian(const coord_3 &max, const coord_3 &min, double resol) {
    this->resol = resol;
    this->coord_max = this->fit_resol(max);
    this->coord_min = this->fit_resol(min);
}

Cartesian::Cartesian(const coord_3 &max, double resol) {
    this->resol = resol;
    this->coord_max = this->fit_resol(max);
    this->coord_min = coord_3(0, 0, 0);
}

point_3 Cartesian::point_at(double x, double y, double z) const {
    return this->fit_resol(coord_3(x, y, z));
}

std::ostream& operator<<(std::ostream& os, Cartesian ct) {
    os << "cartesian coordinate system:" << std::endl;
    os << "   X axis in range (" << ct.min_x() << ", " << ct.max_x() << ")" << std::endl;
    os << "   Y axis in range (" << ct.min_y() << ", " << ct.max_y() << ")" << std::endl;
    os << "   Z axis in range (" << ct.min_z() << ", " << ct.max_z() << ")" << std::endl;
    os << "   resolution: " << ct.resolu() << std::endl;
    return os;
}

std::string Cartesian::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

/**
struct Source {

	virtual double excitation(double t) = 0;

};
**/

struct PointSource {

	PointSource(): index() {

	}

    PointSource(const vector_3 &point, double central_freq, double band_width):
			pos(point), index(), band_width(band_width) {
        omega_central = 2e9 * pi * central_freq;
        tao = sqrt(2.3) / (pi * 0.5e9 * band_width);
    }

#ifdef FDTD_PYTHON_MODULE

    PointSource(PyObject *point, double central_freq, double band_width):
            pos(point), index(), band_width(band_width) {
        omega_central = 2e9 * pi * central_freq;
        tao = sqrt(2.3) / (pi * 0.5e9 * band_width);
    }

#endif

    double excitation(double t) const {
        return sin(omega_central * t) * GaussianPulse(t - 4.5 * tao, tao);
    }

	double bandWidth() const {
		return this->band_width;
	}

	point_3 position() const {
		return this->pos;
	}

	double centralFreq() const {
		return (omega_central / (2 * pi * 1e9));
	}

	void setIndex(const Index3 &index) {
		this->index = index;
	}

private:

	double band_width;
    double omega_central;
    double tao;

	vector_3 pos;
	Index3 index;
};

struct PmlGemp {
    double Exy;
    double Exz;

    double Eyx;
    double Eyz;

    double Ezx;
    double Ezy;

    double Hxy;
    double Hxz;

    double Hyx;
    double Hyz;

    double Hzx;
    double Hzy;

    float sigma_pex;
    float sigma_pey;
    float sigma_pez;

    float sigma_pmx;
    float sigma_pmy;
    float sigma_pmz;
};

struct Gemp { // general electro-magnetic properties

    Gemp(): eps_x(eps_0), eps_y(eps_0), eps_z(eps_0),
            mu_x(mu_0),   mu_y(mu_0),   mu_z(mu_0),
            sig_x(0),     sig_y(0),     sig_z(0),   pml(0),
            f1(0), f2(0), f3(0), f4(0) {

    }

    vector_3 eps() const {
        return vector_3(eps_x, eps_y, eps_z);
    }

    vector_3 mu() const {
        return vector_3(mu_x, mu_y, mu_z);
    }

    vector_3 sig() const {
        return vector_3(sig_x, sig_y, sig_z);
    }

    void assign_pml() {
        if(pml == 0) {
            pml = new PmlGemp();
        }
    }

    void release_pml() {
        if(pml != 0) {
            delete pml;
            pml = 0;
        }
    }

    ~Gemp() {
        release_pml();
    }

    vector_3 E;
    vector_3 H;


    float eps_x;
    float eps_y;
    float eps_z;

    float mu_x;
    float mu_y;
    float mu_z;

    float sig_x;
    float sig_y;
    float sig_z;

    PmlGemp *pml;

    char f1;
    char f2;
    char f3;
    char f4;

};


std::ostream &operator<<(std::ostream &os, const Gemp &phv) {
    os << "eps_x: " << phv.eps_x << std::endl;
    os << "eps_y: " << phv.eps_y << std::endl;
    os << "eps_z: " << phv.eps_z << std::endl;

    os << "mu_x: " << phv.mu_x << std::endl;
    os << "mu_y: " << phv.mu_y << std::endl;
    os << "mu_z: " << phv.mu_z << std::endl;

    os << "sig_x: " << phv.sig_x << std::endl;
    os << "sig_y: " << phv.sig_y << std::endl;
    os << "sig_z: " << phv.sig_z << std::endl;

    return os;
}



class Signal {

public:

    Signal() {

    }

    Signal(const vector_3 &position, double time_pace, double time_begin)
        : posi(position), dt(time_pace), time_begin(time_begin), sigv() {
    }

    Signal(const vector_3 &position, double time_pace)
        : posi(position), dt(time_pace), time_begin(0), sigv() {

    }

#ifdef FDTD_PYTHON_MODULE

    Signal(PyObject *position, double time_pace, double time_begin)
        : posi(position), dt(time_pace), time_begin(time_begin), sigv() {
    }

    Signal(PyObject *position, double time_pace)
        : posi(position), dt(time_pace), time_begin(0), sigv() {

    }

#endif

    Signal subSignal(unsigned low, unsigned high) {
        Signal sig(posi, dt, time_begin + low * dt);
        for(unsigned i = low; i < high; i++) {
            sig.pushValue(this->at(i));
        }
        return sig;
    }

    void pushValue(double vl) {
        sigv.push_back(vl);
    }

	void setTimePace(double pace) {
		this->dt = pace;
	}

    void setIndex(const Index3 &index) {
        this->index = index;
    }

    double at(unsigned i) const {
        return sigv.at(i);
    }

    double power() const {
        double p = 0;
        for(auto iter = sigv.begin(); iter != sigv.end(); ++iter) {
            p += pow(*iter, 2) * dt;
        }
        return p;
    }

	Signal trim(double eps) {
		int end;
		for(int i = this->length() - 1; i >= 0; i--) {
            //std::cout << i << std::endl;
			if(fabs(this->at(i)) > eps) {
				end = i + 1;
                break;
			}
		}
        //std::cout << end << std::endl;
		return this->subSignal(0, end);
	}

    double meanExcessDelay() {
        int f_delay;
        double total = 0, den = 0, weigh = 0;
        double tau_m;
        for(int i = 0; i < this->length(); i++) {
            if(this->at(i) != 0) {
                f_delay = i;
                break;
            }
        }
        //std::cout << f_delay << std::endl;
        for(int i = f_delay; i < this->length(); i++) {
            weigh = pow(this->at(i), 2);
            total += dt * (i - f_delay) * weigh;
            den += weigh;
        }
        return total / den;
    }

    double RMS_DelaySpread() {
        int f_delay;
        double total = 0, den = 0, weigh = 0;
        double tau_m;
        for(int i = 0; i < this->length(); i++) {
            if(this->at(i) != 0) {
                f_delay = i;
                break;
            }
        }
        //std::cout << f_delay << std::endl;
        for(int i = f_delay; i < this->length(); i++) {
            weigh = pow(this->at(i), 2);
            total += dt * (i - f_delay) * weigh;
            den += weigh;
        }
        tau_m = total / den;
        total = 0;
        den = 0;
        for(int i = f_delay; i < this->length(); i++) {
            weigh = pow(this->at(i), 2);
            total += pow((dt * (i - f_delay) - tau_m), 2) * weigh;
            den += weigh;
        }
        return sqrt(total / den);
    }

    int multipathNumber_10dB() {
        double peak = 0;
        double count = 0;
        double max = 0;
        double power_10db;
        double power_t = 0;
        std::vector<int> mark;
        int max_i;
        for(int i = 0; i < this->length(); i++) {
            if(this->at(i) > peak) {
                peak = this->at(i);
            }
        }
        power_10db = 10 * peak;
        //std::cout << power_10db << std::endl;
        while(power_t < power_10db) {
            for(int i = 0; i < this->length(); i++) {
                if(this->at(i) > max && this->at(i) < peak) {
                    max = this->at(i);
                    max_i = i;
                }
                else if(this->at(i) == peak && std::count(mark.begin(), mark.end(), i) == 0) {
                    max = this->at(i);
                    max_i = i;
                }
            }
            peak = max;
            mark.push_back(max_i);
            power_t = power_t + max;
            max = 0;
            //std::cout << power_t << std::endl;
            count ++;
        }
        return count - 1;
    } 

	Signal parseCIR(const Signal &pulse, double eps) {
		int p = 0, n = 0;
		double p_max = 0.0, s_max = 0.0, coef;

		Signal cir(this->posi, this->dt, this->time_begin);
		Signal temp(this->posi, this->dt, this->time_begin);

		for(int i = 0; i < this->length(); i++) {
			cir.pushValue(0.0);
			temp.pushValue(this->at(i));
		}
		for(int i = 0; i < pulse.length(); i++) {
			if(fabs(pulse.at(i)) > fabs(p_max)) {
				p = i;
				p_max = pulse.at(p);
			}
		}
        //std::cout << p << ": " << p_max << std::endl;
		do {
            s_max = 0;
			for(int i = 0; i < temp.length(); i++) {
				if(fabs(temp.at(i)) > fabs(s_max)) {
					n = i;
					s_max = temp.at(i);
				}
			}
            //std::cout << n << ": " << s_max << std::endl;
			coef = s_max / p_max;
            //std::cout << n << ": " << s_max << std::endl;
			cir.sigv[n] = coef;
			for(int i = n - p; i < n + pulse.length() - p; i++)
				if(i >= 0 && i < this->length()) {
					temp.sigv[i] = temp.sigv[i] - coef * pulse.at(p + i - n);
				}
		} while(fabs(s_max) > eps);

		return cir;
	}
    //double at(double t) const {
    //  unsigned i = round((t - time_begin) / dt);
    //  return this->at(i);
    //}

    double timeBegin() const {
        return time_begin;
    }

    double timeEnd() const {
        return time_begin + dt * sigv.size();
    }

    double timePace() const {
        return dt;
    }

    bool timeValid(double t) const {
        return (t >= time_begin && t <= timeEnd());
    }

    double timeValue(double t) const {
        return sigv.at((int)((t - time_begin) / dt));
    }

    vector_3 position() const {
        return posi;
    }

    const Index3 &index3() const {
        return index;
    }

    unsigned length() const {
        return sigv.size();
    }

    void save(std::string filename) const;


	static Signal load(std::string filename);

    std::string str() const;

private:

    std::vector<double> sigv;
    vector_3 posi;
    Index3 index;

    double dt;
    double time_begin;

};

std::ostream &operator<<(std::ostream &os, const Signal &sig) {
    double time_begin = sig.timeBegin();
	int length = sig.length();
    for(unsigned i = 0; i < length; i++) {
        os << time_begin + i * sig.timePace() << "\t";
        os << sig.at(i);
		if(i != length - 1)
			os << std::endl;
    }
    return os;
}

void Signal::save(std::string filename) const {
    std::fstream fs(filename, std::ios::out);
    fs << *this << std::endl;
    fs.close();
}

Signal Signal::load(std::string filename) {
	double t, amp;
	std::fstream fs(filename, std::ios::in);
	std::vector<double> time;
	Signal sig;
	while(!fs.eof()) {
		fs >> t >> amp;
		time.push_back(t);
		sig.pushValue(amp);
	}
	sig.setTimePace(time.at(1) - time.at(0));
    return sig;
}

std::string Signal::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

class Fdtd {

public:

    Fdtd(): ds(0.0), dt(0.0), pml_width(0.0),
        Nx(0), Ny(0), Nz(0), Nt(0),
        Sx(0), Sy(0), Sz(0),
        time_beg(0.0),
        time_end(0.0) {

    }

    void setCoordinateSystem(Cartesian cartesian) {
        this->cart = cartesian;

        this->ds = cart.resolu();
        this->Nx = (cart.max_x() - cart.min_x()) / ds + 1;
        this->Ny = (cart.max_y() - cart.min_y()) / ds + 1;
        this->Nz = (cart.max_z() - cart.min_z()) / ds + 1;

        this->Sx = Nx + 2 * PML::TRUNCATE;
        this->Sy = Ny + 2 * PML::TRUNCATE;
        this->Sz = Nz + 2 * PML::TRUNCATE;

        this->pml_width = PML::THICKNESS * ds;

        this->prepareMemory();
    }

    void setTimePace(double dt) {
        this->dt = dt;
    }

    double Ex(unsigned i, unsigned j, unsigned k) const {
        return this->phy(Index3(i, j, k)).E.x;
    }

    double Ey(unsigned i, unsigned j, unsigned k) const {
        return this->phy(Index3(i, j, k)).E.y;
    }

    double Ez(unsigned i, unsigned j, unsigned k) const {
        return this->phy(Index3(i, j, k)).E.z;
    }

    double Hx(unsigned i, unsigned j, unsigned k) const {
        return this->phy(Index3(i, j, k)).H.x;
    }

    double Hy(unsigned i, unsigned j, unsigned k) const {
        return this->phy(Index3(i, j, k)).H.y;
    }

    double Hz(unsigned i, unsigned j, unsigned k) const {
        return this->phy(Index3(i, j, k)).H.z;
    }

    Gemp &Gp(unsigned i, unsigned j, unsigned k) {
        return this->phy(Index3(i, j, k));
    }


#ifndef __ABC_PML__
#define __ABC_PML__

#endif

    void updateEField() {

        for(unsigned i = 0; i < Sx; i++)
            for(unsigned j = 1; j < Sy; j++)
                for(unsigned k = 1; k < Sz; k++) {

                    Gemp &gp = this->Gp(i, j, k);

                    if(gp.f1 == PML::REGION_FREE) {
                        double Cexe = (2 * gp.eps_x - dt * gp.sig_x) / (2 * gp.eps_x + dt * gp.sig_x);
                        double Cexhz = (2 * dt) / ((2 * gp.eps_x + dt * gp.sig_x) * ds);
                        double Cexhy = - Cexhz; // dz = dy = ds

                        gp.E.x = Cexe * gp.E.x
                                + Cexhz * (gp.H.z - Hz(i, j-1, k))
                                + Cexhy * (gp.H.y - Hy(i, j, k-1))
                            ;
                    }
#ifdef __ABC_PML__
                    else {

                        double sigma_pey = gp.pml->sigma_pey;
                        double sigma_pez = gp.pml->sigma_pez;
                        
                        double Cexye = (2 * gp.eps_x - dt * sigma_pey) / (2 * gp.eps_x + dt * sigma_pey);
                        double Cexhz = (2 * dt) / ((2 * gp.eps_x + dt * sigma_pey) * ds);
                        gp.pml->Exy = Cexye * gp.pml->Exy + Cexhz * (gp.H.z - Hz(i, j-1, k));
                        
                        double Cexze = (2 * gp.eps_x - dt * sigma_pez) / (2 * gp.eps_x + dt * sigma_pez);
                        double Cexhy = - (2 * dt) / ((2 * gp.eps_x + dt * sigma_pez) * ds);
                        gp.pml->Exz = Cexze * gp.pml->Exz + Cexhy * (gp.H.y - Hy(i, j, k-1));

                        gp.E.x = gp.pml->Exy + gp.pml->Exz;

                    }
#endif
                }

        for(unsigned i = 1; i < Sx; i++)
            for(unsigned j = 0; j < Sy; j++)
                for(unsigned k = 1; k < Sz; k++) {

                    Gemp &gp = this->Gp(i, j, k);

                    if(gp.f1 == PML::REGION_FREE) {
                        double Ceye = (2 * gp.eps_y - dt * gp.sig_y) / (2 * gp.eps_y + dt * gp.sig_y);
                        double Ceyhx = (2 * dt) / ((2 * gp.eps_y + dt * gp.sig_y) * ds);
                        double Ceyhz = - Ceyhx;

                        gp.E.y = Ceye * gp.E.y
                                + Ceyhx * (gp.H.x - Hx(i, j, k-1))
                                + Ceyhz * (gp.H.z - Hz(i-1, j, k))
                            ;
                    }
#ifdef __ABC_PML__
                    else {

                        double sigma_pex = gp.pml->sigma_pex;
                        double sigma_pez = gp.pml->sigma_pez;

                        double Ceyxe = (2 * gp.eps_y - dt * sigma_pex) / (2 * gp.eps_y + dt * sigma_pex);
                        double Ceyhz = - (2 * dt) / ((2 * gp.eps_y + dt * sigma_pex) * ds);
                        gp.pml->Eyx = Ceyxe * gp.pml->Eyx + Ceyhz * (gp.H.z - Hz(i-1, j, k));

                        double Ceyze = (2 * gp.eps_y - dt * sigma_pez) / (2 * gp.eps_y + dt * sigma_pez);
                        double Ceyhx = (2 * dt) / ((2 * gp.eps_y + dt * sigma_pez) * ds);
                        gp.pml->Eyz = Ceyze * gp.pml->Eyz + Ceyhx * (gp.H.x - Hx(i, j, k-1));

                        gp.E.y = gp.pml->Eyx + gp.pml->Eyz;

                    }
#endif
                }

        for(unsigned i = 1; i < Sx; i++)
            for(unsigned j = 1; j < Sy; j++)
                for(unsigned k = 0; k < Sz; k++) {

                    Gemp &gp = this->Gp(i, j, k);

                    if(gp.f1 == PML::REGION_FREE) {
                        double Ceze = (2 * gp.eps_z - dt * gp.sig_z) / (2 * gp.eps_z + dt * gp.sig_z);
                        double Cezhy = (2 * dt) / ((2 * gp.eps_z + dt * gp.sig_z) * ds);
                        double Cezhx = - Cezhy;

                        gp.E.z = Ceze * gp.E.z
                                + Cezhy * (gp.H.y - Hy(i-1, j, k))
                                + Cezhx * (gp.H.x - Hx(i, j-1, k))
                            ;
                    }
#ifdef __ABC_PML__
                    else {

                        double sigma_pex = gp.pml->sigma_pex;
                        double sigma_pey = gp.pml->sigma_pey;

                        
                        double Cezxe = (2 * gp.eps_z - dt * sigma_pex) / (2 * gp.eps_z + dt * sigma_pex);
                        double Cezhy = (2 * dt) / ((2 * gp.eps_z + dt * sigma_pex) * ds);
                        gp.pml->Ezx = Cezxe * gp.pml->Ezx + Cezhy * (gp.H.y - Hy(i-1, j, k));
                        


                        double Cezye = (2 * gp.eps_z - dt * sigma_pey) / (2 * gp.eps_z + dt * sigma_pey);
                        double Cezhx = - (2 * dt) / ((2 * gp.eps_z + dt * sigma_pey) * ds);
                        gp.pml->Ezy = Cezye * gp.pml->Ezy + Cezhx * (gp.H.x - Hx(i, j-1, k));


                        gp.E.z = gp.pml->Ezx + gp.pml->Ezy;
                    }
#endif
                }
    }

    void updateHField() {

        for(unsigned i = 0; i < Sx; i++)
            for(unsigned j = 0; j < Sy - 1; j++)
                for(unsigned k = 0; k < Sz - 1; k++) {

                    Gemp &gp = this->Gp(i, j, k);

                    if(gp.f1 == PML::REGION_FREE) {
                        double Chxh = 1.0;
                        double Chxey = (2 * dt) / (2 * gp.mu_x * ds);
                        double Chxez = - Chxey;

                        gp.H.x = Chxh * gp.H.x
                                + Chxey * (Ey(i, j, k+1) - gp.E.y)
                                + Chxez * (Ez(i, j+1, k) - gp.E.z)
                            ;
                    }
#ifdef __ABC_PML__
                    else {

                        double sigma_pmy = gp.pml->sigma_pmy;
                        double sigma_pmz = gp.pml->sigma_pmz;


                        double Chxyh = (2 * gp.mu_x - dt * sigma_pmy) / (2 * gp.mu_x + dt * sigma_pmy);
                        double Chxez = - (2 * dt) / ((2 * gp.mu_x + dt * sigma_pmy) * ds);
                        gp.pml->Hxy = Chxyh * gp.pml->Hxy + Chxez * (Ez(i, j+1, k) - gp.E.z);


                        double Chxzh = (2 * gp.mu_x - dt * sigma_pmz) / (2 * gp.mu_x + dt * sigma_pmz);
                        double Chxey = (2 * dt) / ((2 * gp.mu_x + dt * sigma_pmz) * ds);
                        gp.pml->Hxz = Chxzh * gp.pml->Hxz + Chxey * (Ey(i, j, k+1) - gp.E.y);


                        gp.H.x = gp.pml->Hxy + gp.pml->Hxz;

                    }
#endif
                }

        for(unsigned i = 0; i < Sx - 1; i++)
            for(unsigned j = 0; j < Sy; j++)
                for(unsigned k = 0; k < Sz - 1; k++) {

                    Gemp &gp = this->Gp(i, j, k);

                    if(gp.f1 == PML::REGION_FREE) {
                        double Chyh = 1.0;
                        double Chyez = (2 * dt) / (2 * gp.mu_y * ds);
                        double Chyex = - Chyez;

                        gp.H.y = Chyh * gp.H.y
                                + Chyez * (Ez(i+1, j, k) - gp.E.z)
                                + Chyex * (Ex(i, j, k+1) - gp.E.x)
                            ;
                    }
#ifdef __ABC_PML__
                    else {

                        double sigma_pmx = gp.pml->sigma_pmx;
                        double sigma_pmz = gp.pml->sigma_pmz;

                        double Chyzh = (2 * gp.mu_y - dt * sigma_pmz) / (2 * gp.mu_y + dt * sigma_pmz);
                        double Chyex =  - (2 * dt) / ((2 * gp.mu_y + dt * sigma_pmz) * ds);
                        gp.pml->Hyz = Chyzh * gp.pml->Hyz + Chyex * (Ex(i, j, k+1) - gp.E.x);

                        double Chyxh = (2 * gp.mu_y - dt * sigma_pmx) / (2 * gp.mu_y + dt * sigma_pmx);
                        double Chyez =  (2 * dt) / ((2 * gp.mu_y + dt * sigma_pmx) * ds);
                        gp.pml->Hyx = Chyxh * gp.pml->Hyx + Chyez * (Ez(i+1, j, k) - gp.E.z);

                        gp.H.y = gp.pml->Hyx + gp.pml->Hyz;

                    }
#endif
                }

        for(unsigned i = 0; i < Sx - 1; i++)
            for(unsigned j = 0; j < Sy - 1; j++)
                for(unsigned k = 0; k < Sz; k++) {

                    Gemp &gp = this->Gp(i, j, k);

                    if(gp.f1 == PML::REGION_FREE) {
                        double Chzh = 1.0;
                        double Chzex = (2 * dt) / (2 * gp.mu_z * ds);
                        double Chzey = - Chzex;

                        gp.H.z = Chzh * gp.H.z
                                + Chzex * (Ex(i, j+1, k) - gp.E.x)
                                + Chzey * (Ey(i+1, j, k) - gp.E.y)
                            ;
                    }
#ifdef __ABC_PML__
                    else {

                        double sigma_pmx = gp.pml->sigma_pmx;
                        double sigma_pmy = gp.pml->sigma_pmy;

                        double Chzyh = (2 * gp.mu_z - dt * sigma_pmy) / (2 * gp.mu_z + dt * sigma_pmy);
                        double Chzex = (2 * dt) / ((2 * gp.mu_z + dt * sigma_pmy) * ds);
                        gp.pml->Hzy = Chzyh * gp.pml->Hzy + Chzex * (Ex(i, j+1, k) - gp.E.x);

                        double Chzxh = (2 * gp.mu_z - dt * sigma_pmx) / (2 * gp.mu_z + dt * sigma_pmx);
                        double Chzey = - (2 * dt) / ((2 * gp.mu_z + dt * sigma_pmx) * ds);
                        gp.pml->Hzx = Chzxh * gp.pml->Hzx + Chzey * (Ey(i+1, j, k) - gp.E.y);

                        gp.H.z = gp.pml->Hzx + gp.pml->Hzy;

                    }
#endif
                }
    }

    void updateSource(double t) {
        //static double tao = 350 * dt;
        //for(auto iter = dipole_source.begin(); iter != dipole_source.end(); ++iter) {
        //    for(unsigned i = iter->Izb; i < iter->Ize; i++) {
        //        this->phy(Index3(iter->Ix, iter->Iy, i)).E.z = 0;
        //    }
        //    // this->phy(Index3(iter->Ix, iter->Iy, iter->Izc)).E.z = 100.0 * dd_gaussian((t - 3 * tao) * 1e17, tao);
        //    this->phy(Index3(iter->Ix, iter->Iy, iter->Izc)).E.z = iter->excitation(t);
        //}
		for(auto iter = point_source.begin(); iter != point_source.end(); ++iter) {
            this->phy(iter->index).E.z = iter->excitation(t);
		}
    }

    void updateReceive() {
        for(auto iter = recvs.begin(); iter != recvs.end(); ++iter) {
            const Gemp &gp = this->space(iter->index3());
            iter->pushValue(gp.E.z);
        }
    }

    void promote(double t) { // update formulas

        if(t > time_end) {
            time_end = t;
            Nt = round(time_end / dt) + 1;

            std::cout << "Time Begin: " << this->time_beg << "s" << std::endl;
            std::cout << "Time End:   " << this->time_end << "s" << std::endl;
            std::cout << "Time Pace:  " << this->dt << "s" << std::endl;
            std::cout << "Steps:   " << Nt - 1 << std::endl;

            for(unsigned nt = time_beg / dt; nt < Nt; nt++) {

#ifdef FDTD_PYTHON_MODULE

                PySys_WriteStdout("%5d / %5d\n", nt, Nt-1);

#else

                printf("%5d / %5d\n", nt, Nt-1);

#endif
                double t = dt * nt;
                updateHField();
                updateEField();
                updateSource(t);
                updateReceive();
            }
            time_beg = Nt * dt;
        }
    }

    void prepareMemory() {

        std::cout << "Nx: " << this->Nx << std::endl;
        std::cout << "Ny: " << this->Ny << std::endl;
        std::cout << "Nz: " << this->Nz << std::endl;
        //std::cout << "Nt: " << this->Nt << std::endl;

        std::cout << this->cart << std::endl;

        Assert(ProperDivide(cart.max_x(), cart.min_x(), ds), "Coordinate system error...");
        Assert(ProperDivide(cart.max_y(), cart.min_y(), ds), "Coordinate system error...");
        Assert(ProperDivide(cart.max_z(), cart.min_z(), ds), "Coordinate system error...");
        //Assert(ProperDivide(time_end, time_beg, dt), "Time setting error...");

        this->phy.resize(Sx, Sy, Sz);
        this->space.reference(this->phy(blitz::Range(PML::TRUNCATE, PML::TRUNCATE + Nx - 1),
                                            blitz::Range(PML::TRUNCATE, PML::TRUNCATE + Ny - 1),
                                            blitz::Range(PML::TRUNCATE, PML::TRUNCATE + Nz - 1))
                );

        this->preparePMLLayer();
    }

    void preparePMLLayer() {
        std::cout << "Preparing PML Layer..." << std::endl;
        for(unsigned i = 0; i < PML::THICKNESS; i++)
            for(unsigned j = 0; j < Sy; j++)
                for(unsigned k = 0; k < Sz; k++) {

                    Gemp &xn = this->phy(Index3(i, j, k));
                    xn.f1 |= PML::REGION_XNP;
                    xn.assign_pml();
                    xn.pml->sigma_pex = this->getSigmaPe(pml_width - ds * i - ds / 2);
                    xn.pml->sigma_pmx = this->getSigmaPm(pml_width - ds * i);

                    Gemp &xp = this->phy(Index3(Sx - 1 - i, j, k));
                    xp.f1 |= PML::REGION_XNP;
                    xp.assign_pml();
                    xp.pml->sigma_pex = this->getSigmaPe(pml_width - ds * i);
                    xp.pml->sigma_pmx = this->getSigmaPm(pml_width - ds * i - ds / 2);

                }
        for(unsigned i = 0; i < Sx; i++)
            for(unsigned j = 0; j < PML::THICKNESS; j++)
                for(unsigned k = 0; k < Sz; k++) {

                    Gemp &yn = this->phy(Index3(i, j, k));
                    yn.f1 |= PML::REGION_YNP;
                    yn.assign_pml();
                    yn.pml->sigma_pey = this->getSigmaPe(pml_width - ds * j - ds / 2);
                    yn.pml->sigma_pmy = this->getSigmaPm(pml_width - ds * j);

                    Gemp &yp = this->phy(Index3(i, Sy - 1 - j, k));
                    yp.f1 |= PML::REGION_YNP;
                    yp.assign_pml();
                    yp.pml->sigma_pey = this->getSigmaPe(pml_width - ds * j);
                    yp.pml->sigma_pmy = this->getSigmaPm(pml_width - ds * j - ds / 2);

                }
        for(unsigned i = 0; i < Sx; i++)
            for(unsigned j = 0; j < Sy; j++)
                for(unsigned k = 0; k < PML::THICKNESS; k++) {

                    Gemp &zn = this->phy(Index3(i, j, k));
                    zn.f1 |= PML::REGION_ZNP;
                    zn.assign_pml();
                    zn.pml->sigma_pez = this->getSigmaPe(pml_width - ds * k - ds / 2);
                    zn.pml->sigma_pmz = this->getSigmaPm(pml_width - ds * k);

                    Gemp &zp = this->phy(Index3(i, j, Sz - 1 - k));
                    zp.f1 |= PML::REGION_ZNP;
                    zp.assign_pml();
                    zp.pml->sigma_pez = this->getSigmaPe(pml_width - ds * k);
                    zp.pml->sigma_pmz = this->getSigmaPm(pml_width - ds * k - ds / 2);

                }

    }

    void release() {
        this->Nt = 0;
        this->Nx = 0, this->Ny = 0, this->Nz = 0;
        this->cart = Cartesian();
        this->dipole_source.clear();
        this->ds = 0.0;
        this->phy.free();
        this->space.free();
        this->time_beg = 0.0;
        this->time_end = 0.0;
        this->pml_width = 0.0;
    }

    double cflCondition() const {
        return ds / (c * sqrt(3.0));
    }

    void testCflCondition() const {
        if(dt >= this->cflCondition()) {
            std::cout << "Time step doesn't satisfy CFL condition..." << std::endl;
            exit(1);
        }
    }

    void addDipoleSource(point_3 pt, double arm_length, double central_freq, double band_width) {
        int xi = round((pt.x - cart.min_x()) / ds) + PML::TRUNCATE;
        int yi = round((pt.y - cart.min_y()) / ds) + PML::TRUNCATE;
        int zc = round((pt.z - cart.min_z()) / ds) + PML::TRUNCATE;
        int zl = round(arm_length / ds);

        if(arm_length == inf) {
            this->dipole_source.push_back(Dipole(xi, yi, 0,
                Sz, central_freq, band_width));
        }
        else {
            this->dipole_source.push_back(Dipole(xi, yi, Max(zc-zl, 0),
                Min(zc+zl+1, Sz), central_freq, band_width));
        }
        //std::cout << zl << std::endl;
        //std::cout << Max(zc-zl, 1) << std::endl;
        //std::cout << Min(zc+zl+1, Sz-1) << std::endl;
    }

#ifdef FDTD_PYTHON_MODULE

    bool addPointSource_Py(PyObject *obj, double central_freq, double band_width) {
        vector_3 pt(obj);
        PointSource ps(pt, central_freq, band_width);
        int xi = round((pt.x - cart.min_x()) / ds) + PML::TRUNCATE;
        int yi = round((pt.y - cart.min_y()) / ds) + PML::TRUNCATE;
        int zi = round((pt.z - cart.min_z()) / ds) + PML::TRUNCATE;
        if(xi < 0 || yi < 0 || zi < 0 || xi >= Sx || yi >= Sy || zi >= Sz) {
            return false;
        }
        ps.setIndex(Index3(xi, yi, zi));
        this->point_source.push_back(ps);
    }

#endif

	bool addPointSource(const point_3 &pt, double central_freq, double band_width) {
		PointSource ps(pt, central_freq, band_width);
		int xi = round((pt.x - cart.min_x()) / ds) + PML::TRUNCATE;
        int yi = round((pt.y - cart.min_y()) / ds) + PML::TRUNCATE;
        int zi = round((pt.z - cart.min_z()) / ds) + PML::TRUNCATE;
		if(xi < 0 || yi < 0 || zi < 0 || xi >= Sx || yi >= Sy || zi >= Sz) {
			return false;
		}
		ps.setIndex(Index3(xi, yi, zi));
		this->point_source.push_back(ps);
	}

    void addMedium(const Shape &shp, Medium med) {
        double max_x = Min(shp.max_x(), cart.max_x());
        double max_y = Min(shp.max_y(), cart.max_y());
        double max_z = Min(shp.max_z(), cart.max_z());

        double min_x = Max(shp.min_x(), cart.min_x());
        double min_y = Max(shp.min_y(), cart.min_y());
        double min_z = Max(shp.min_z(), cart.min_z());

        unsigned bi = floor((min_x - cart.min_x()) / ds);
        unsigned bj = floor((min_y - cart.min_y()) / ds);
        unsigned bk = floor((min_z - cart.min_z()) / ds);

        unsigned ei = ceil((max_x - cart.min_x()) / ds) + 1;
        unsigned ej = ceil((max_y - cart.min_y()) / ds) + 1;
        unsigned ek = ceil((max_z - cart.min_z()) / ds) + 1;

        if(bi >= ei || bj >= ej || bk >= ek) {
            std::cout << "Target field invalid..." << std::endl;
        }

        // std::cout << bi << " " << ei << std::endl;
        // std::cout << bj << " " << ej << std::endl;
        // std::cout << bk << " " << ek << std::endl;


        double vr;
        for(unsigned i = bi; i < ei; i++)
            for(unsigned j = bj; j < ej; j++)
                for(unsigned k = bk; k < ek; k++) {
                    Index3 index;
                    index = i, j, k;
                    Gemp &em = space(index);
                    vector_3 coord(i * ds, j * ds, k * ds);

                    vr = this->shapeCoverRatio(coord + vector_3(ds / 2, 0, 0), shp);
                    em.eps_x = med.eps * vr + em.eps_x * (1.0 - vr);
                    em.sig_x = med.sig * vr + em.sig_x * (1.0 - vr);

                    vr = this->shapeCoverRatio(coord + vector_3(0, ds / 2, 0), shp);
                    em.eps_y = med.eps * vr + em.eps_y * (1.0 - vr);
                    em.sig_y = med.sig * vr + em.sig_y * (1.0 - vr);

                    vr = this->shapeCoverRatio(coord + vector_3(0, 0, ds / 2), shp);
                    em.eps_z = med.eps * vr + em.eps_z * (1.0 - vr);
                    em.sig_z = med.sig * vr + em.sig_z * (1.0 - vr);

                    vr = this->shapeCoverRatio(coord + vector_3(0, ds / 2, ds / 2), shp);
                    em.mu_x = med.mu * vr + em.mu_x * (1.0 - vr);

                    vr = this->shapeCoverRatio(coord + vector_3(ds / 2, 0, ds / 2), shp);
                    em.mu_y = med.mu * vr + em.mu_y * (1.0 - vr);

                    vr = this->shapeCoverRatio(coord + vector_3(ds / 2, ds / 2, 0), shp);
                    em.mu_z = med.mu * vr + em.mu_y * (1.0 - vr);

                    //std::cout << vr << std::endl;

                }
    }

    void setWall(double thick, const Medium &med) {
        this->addMedium(Brick(vector_3(-1, -1, -1), vector_3(thick, inf, inf)), med);
        this->addMedium(Brick(vector_3(-1, -1, -1), vector_3(inf, thick, inf)), med);
		this->addMedium(Brick(vector_3(-1, -1, -1), vector_3(inf, inf, thick)), med);

		this->addMedium(Brick(vector_3(cart.max_x() - thick, -1, -1), vector_3(inf, inf, inf)), med);
		this->addMedium(Brick(vector_3(-1, cart.max_y() - thick, -1), vector_3(inf, inf, inf)), med);
		this->addMedium(Brick(vector_3(-1, -1, cart.max_z() - thick), vector_3(inf, inf, inf)), med);
    }


#ifdef FDTD_PYTHON_MODULE

    void addReceivePoint_Py(PyObject *obj) {
        vector_3 point(obj);
        Signal sig(point, dt);
        unsigned i = round((point.x - cart.min_x()) / ds);
        unsigned j = round((point.y - cart.min_y()) / ds);
        unsigned k = round((point.z - cart.min_z()) / ds);
        sig.setIndex(Index3(i, j, k));

        this->recvs.push_back(sig);
    }

#endif

    void addReceivePoint(const vector_3 &point) {
        Signal sig(point, dt);
        unsigned i = round((point.x - cart.min_x()) / ds);
        unsigned j = round((point.y - cart.min_y()) / ds);
        unsigned k = round((point.z - cart.min_z()) / ds);
        sig.setIndex(Index3(i, j, k));

        this->recvs.push_back(sig);
    }

    void addReceivePoint(double x, double y, double z) {
        this->addReceivePoint(vector_3(x, y, z));
    }

    Signal getReceiveSignal(unsigned i) {
        return this->recvs.at(i);
    }

    PointSource getPointSource(unsigned i) {
        return this->point_source.at(i);
    }

    unsigned getSizeX() const {
        return Sx;
    }

    unsigned getSizeY() const {
        return Sy;
    }

    unsigned getSizeZ() const {
        return Sz;
    }

    unsigned getSizePML() const {
        return PML::THICKNESS;
    }

    double getDs() const {
        return ds;
    }

    double getDt() const {
        return dt;
    }


    template <typename Func>
    bool exportXYDataFile(double z, Func pattern, std::string fname = "data.fde") {

        unsigned zi = round((z - cart.min_z()) / ds);

        if(z > cart.max_z() || z < cart.min_z()) {
            std::cout << "Target field invalid..." << std::endl;
        }

        std::fstream fs(fname, std::ios::out);
        fs << "X, Y" << std::endl;
        fs << 0 << ", " << ds * (Sx - 1) << ", " << ds << std::endl;
        fs << 0 << ", " << ds * (Sy - 1) << ", " << ds << std::endl;

        for(unsigned i = 0; i < Sx; i++) {
            for(unsigned j = 0; j < Sy; j++) {
                fs << pattern(this->phy(Index3(i, j, zi))) << ", ";
            }
            fs << std::endl;
        }
        fs.close();
        return true;
    }


    template <typename Func>
    bool exportYZDataFile(double x, Func pattern, std::string fname = "data.fde") {

        unsigned xi = round((x - cart.min_x()) / ds);

        if(x > cart.max_x() || x < cart.min_x()) {
            std::cout << "Target field invalid..." << std::endl;
        }

        std::fstream fs(fname, std::ios::out);
        fs << "Y, Z" << std::endl;
        fs << 0 << ", " << ds * (Sy - 1) << ", " << ds << std::endl;
        fs << 0 << ", " << ds * (Sz - 1) << ", " << ds << std::endl;

        for(unsigned i = 0; i < Sy; i++) {
            for(unsigned j = 0; j < Sz; j++) {
                fs << pattern(this->phy(xi, i, j)) << ", ";
            }
            fs << std::endl;
        }
        fs.close();
        return true;
    }

    template <typename Func>
    bool exportXZDataFile(double y, Func pattern, std::string fname = "data.fde") {

        unsigned yi = round((y - cart.min_y()) / ds);

        if(y > cart.max_y() || y < cart.min_y()) {
            std::cout << "Target field invalid..." << std::endl;
        }

        std::fstream fs(fname, std::ios::out);
        fs << "Y, Z" << std::endl;
        fs << 0 << ", " << ds * (Sx - 1) << ", " << ds << std::endl;
        fs << 0 << ", " << ds * (Sz - 1) << ", " << ds << std::endl;

        for(unsigned i = 0; i < Sx; i++) {
            for(unsigned j = 0; j < Sz; j++) {
                fs << pattern(this->phy(Index3(i, yi, j))) << ", ";
            }
            fs << std::endl;
        }
        fs.close();
        return true;
    }

private:

    blitz::Array<Gemp, 3> phy;
    blitz::Array<Gemp, 3> space;


    Cartesian cart;

    std::vector<Signal> recvs;

    double time_beg;
    double time_end;

    double ds;
    double dt;

    double pml_width;

    double sigma_pe_max;
    double sigma_pm_max;


    unsigned Nx;
    unsigned Ny;
    unsigned Nz;
    unsigned Nt;

    unsigned Sx;
    unsigned Sy;
    unsigned Sz;

    double shapeCoverRatio(const point_3 &pt, const Shape &shp) {

        static vector_3 sub_cube_coff[8] = {
            vector_3(ds / 4, ds / 4, ds / 4),
            vector_3(-ds / 4, ds / 4, ds / 4),
            vector_3(ds / 4, -ds / 4, ds / 4),
            vector_3(ds / 4, ds / 4, -ds / 4),
            vector_3(-ds / 4, -ds / 4, ds / 4),
            vector_3(ds / 4, -ds / 4, -ds / 4),
            vector_3(-ds / 4, ds / 4, -ds / 4),
            vector_3(-ds / 4, -ds / 4, -ds / 4),
        };

        unsigned count = 0;
        for(int i = 0; i < 8; i++) {
            if(shp.hasInnerPoint(pt + sub_cube_coff[i]))
                count++;
        }
        return count / 8.0;

    }

    //static const std::string OUTPUT_DIV = ", ";
    struct Dipole {
        Dipole(unsigned Ix, unsigned Iy, unsigned Izb, unsigned Ize, double central_freq, double band_width):
            Ix(Ix), Iy(Iy), Izb(Izb), Ize(Ize) {

            Izc = (Izb + Ize) / 2;
            omega_central = 2e9 * pi * central_freq;
            tao = sqrt(2.3) / (pi * 0.5e9 * band_width);
        }

        double excitation(double t) {
            return sin(omega_central * t) * GaussianPulse(t - 4.5 * tao, tao);
        }

        unsigned Ix;
        unsigned Iy;
        unsigned Izb;
        unsigned Ize;
        unsigned Izc;

        double omega_central;
        double tao;
    };

    std::vector<Dipole> dipole_source;
	std::vector<PointSource> point_source;

#ifdef __GNUG__

    struct PML {

        constexpr static int THICKNESS = 12;
        constexpr static int N_PML = 3;
		constexpr static int TRUNCATE = THICKNESS + 0;

        constexpr static double SIGMA_PE = 1.0;
        constexpr static double SIGMA_PM = (mu_0 / eps_0) * SIGMA_PE;

        constexpr static char REGION_FREE = 0x00;
        constexpr static char REGION_XNP = 0x01;
        constexpr static char REGION_YNP = 0x02;
        constexpr static char REGION_ZNP = 0x04;

    };

#elif _WIN32

    struct PML {

        const static int THICKNESS = 12;
		const static int TRUNCATE = THICKNESS + 0;
        const static int N_PML = 4;

        const static double SIGMA_PE;
        const static double SIGMA_PM;

        const static char REGION_FREE = 0x00;
        const static char REGION_XNP = 0x01;
        const static char REGION_YNP = 0x02;
        const static char REGION_ZNP = 0x04;

    };

#endif

    double getSigmaPeMax() {
		return (PML::N_PML + 1) / (150 * pi * ds);
        // log(1e-7) * ((PML::N_PML + 1) * eps_0 * c) / (2 * ds * PML::THICKNESS);
    }

    double getSigmaPmMax() {
        return (mu_0 / eps_0) * getSigmaPeMax();
    }

    double getSigmaPe(double rho) {
         return getSigmaPeMax() * pow(rho / this->pml_width, PML::N_PML);
    }

    double getSigmaPm(double rho) {
         return getSigmaPmMax() * pow(rho / this->pml_width, PML::N_PML);
    }
    

};

#ifdef _WIN32

const double Fdtd::PML::SIGMA_PE = 1.0;
const double Fdtd::PML::SIGMA_PM = (mu_0 / eps_0) * Fdtd::PML::SIGMA_PE;

#endif

} // end of namespace fdtd

using namespace std;
using namespace fdtd;
using namespace blitz;

int main()
{
    Fdtd task;
    Cartesian ct(vector_3(5, 5, 5), 0.1);

    task.setCoordinateSystem(ct);
    task.setTimePace(1e-11);

    Medium medium;
    medium.sig = 0.001;
    task.setWall(1, medium);

    //cout << task.cpmlSigmaE(0.1) << endl;


    //Brick brick(vector_3(3, 3, 0), vector_3(4, 4, 5));
    //Medium med;
    //med.sig = inf;
    //task.addMedium(brick, med);

    task.addDipoleSource(point_3(2.5, 2.5, 2.5), inf, 0.25, 0.05);
    //task.addReceivePoint(point_3(2.0, 2.0, 2.5));
    task.promote(1e-11);

    //task.getReceiveSignal(0).save("signal.txt");

    task.exportXYDataFile(2.5,
            [](const Gemp &gp) -> double {
                return gp.sig_x;
            });

}



#ifdef FDTD_PYTHON_MODULE


#define BOOST_PYTHON_STATIC_LIB

#include <boost/python.hpp>

using namespace boost::python;
using namespace blitz;
using namespace std;
using namespace fdtd;

struct ShapeWrap: Shape, wrapper<Shape> {

    bool hasInnerPoint(point_3 pt) const {
        return this->get_override("hasInnerPoint")(pt);
    }

    double max_x() const {
        return this->get_override("max_x")();
    }

    double max_y() const {
        return this->get_override("max_y")();
    }

    double max_z() const {
        return this->get_override("max_z")();
    }

    double min_x() const {
        return this->get_override("min_x")();
    }

    double min_y() const {
        return this->get_override("min_y")();
    }

    double min_z() const {
        return this->get_override("min_z")();
    }

    //string shape_name() const {
    //  return this->get_override("shape_name")();
    //}
};


BOOST_PYTHON_MODULE(fdtd_W)
{
    class_<vector_3>("vector_3")
        .def(init<double, double, double>(args("x", "y", "z")))
        .def(init<PyObject *>(args("tuple")))
        .def("tuple", &vector_3::tuple)
        .def_readwrite("x", &vector_3::x)
        .def_readwrite("y", &vector_3::y)
        .def_readwrite("z", &vector_3::z)
        .def(self + self)
        .def(self + other<PyObject *>())
        .def(other<PyObject *>() + self)
        .def(self - self)
        .def(self - other<PyObject *>())
        .def(other<PyObject *>() - self)
        .def(self * double())
        .def(self / double())
        .def(double() * self)
        .def("__str__", &vector_3::str)
    ;

    def("inner_product", fdtd::inner_product, args("a", "b"));
    def("cross_product", fdtd::cross_product, args("a", "b"));
    def("norm", fdtd::norm);
    // def("coord_3", coord_3);
    // def("point_3", point_3);

    //def("pi"   , fdtd::pi);
    //def("eps_0", fdtd::eps_0);
    //def("mu_0" , fdtd::mu_0);
    //def("c"    , fdtd::c);

    //def("VEC_ORIGIN", fdtd::VEC_ORIGIN)
    //def("VEC_UNIT_X", fdtd::VEC_UNIT_X)
    //def("VEC_UNIT_Y", fdtd::VEC_UNIT_Y)
    //def("VEC_UNIT_Z", fdtd::VEC_UNIT_Z)

    class_<Cartesian>("Cartesian")
        //.def(init<const vector_3 &, double>(args("upper_bound", "resol")))
        .def(init<PyObject *, double>(args("upper_bound", "resol")))
        //.def("point_at", &Cartesian::point_at)
        //.def("set_min", &Cartesian::set_min)
        //.def("set_max", &Cartesian::set_max)
        //.def("set_resol", &Cartesian::set_resol)
        .add_property("max_x", &Cartesian::max_x)
        .add_property("max_y", &Cartesian::max_y)
        .add_property("max_z", &Cartesian::max_z)
        .add_property("min_x", &Cartesian::min_x)
        .add_property("min_y", &Cartesian::min_y)
        .add_property("min_z", &Cartesian::min_z)
        .add_property("resol", &Cartesian::resolu)
        .def("__str__", &Cartesian::str)
    ;

    def("gaussian_pulse", fdtd::GaussianPulse, args("time", "tao"));
    def("sin_wave", fdtd::SinWave, args("time", "omega"));

    class_<Medium>("Medium")
        .def(init<>())
        .def(init<double, double, double>(args("eps", "mu", "sig")))
        .def_readwrite("eps", &Medium::eps)
        .def_readwrite("mu", &Medium::mu)
        .def_readwrite("sig", &Medium::sig)
    ;

    class_<ShapeWrap, boost::noncopyable>("Shape")
        .def("hasInnerPoint", pure_virtual(&Shape::hasInnerPoint))
        .def("max_x", pure_virtual(&Shape::max_x))
        .def("max_y", pure_virtual(&Shape::max_y))
        .def("max_z", pure_virtual(&Shape::max_z))
        .def("min_x", pure_virtual(&Shape::min_x))
        .def("min_y", pure_virtual(&Shape::min_y))
        .def("min_z", pure_virtual(&Shape::min_z))
        //.def("shape_name", pure_virtual(&Shape::shape_name))
    ;

    class_<Brick, bases<Shape> >("Brick")
        .def(init<>())
        //.def(init<const vector_3 &, const vector_3 &>(args("v_min", "v_max")))
        .def(init<PyObject *, PyObject *>(args("v_min", "v_max")))
		.add_property("max_x", &Brick::max_x)
		.add_property("max_y", &Brick::max_y)
		.add_property("max_z", &Brick::max_z)
		.add_property("min_x", &Brick::min_x)
		.add_property("min_y", &Brick::min_y)
		.add_property("min_z", &Brick::min_z)
        .def("__str__", &Brick::str)
    ;

    class_<Sphere, bases<Shape> >("Sphere")
        .def(init<>())
        //.def(init<const vector_3 &, double>(args("center", "radius")))
        .def(init<PyObject *, double>(args("center", "radius")))
        .add_property("center", &Sphere::getCenter)
        .add_property("radius", &Sphere::getRadius)
        .def("__str__", &Sphere::str)
    ;


    class_<Gemp>("Gemp")
        .def(init<>())
        .add_property("eps", &Gemp::eps)
        .add_property("mu" , &Gemp::mu)
        .add_property("sig", &Gemp::sig)
        .add_property("E", &Gemp::E)
        .add_property("H", &Gemp::H)
        //.def(str(self))
    ;

	class_<PointSource>("Source")
		//.def(init<const vector_3 &, double, double>())
        .def(init<PyObject *, double, double>())
		.add_property("position", &PointSource::position)
		.add_property("central_freq", &PointSource::centralFreq)
		.add_property("band_width", &PointSource::bandWidth)
		.def("excitation", &PointSource::excitation)
	;

    class_<Signal>("Signal")
        .def(init<>())
        .def(init<PyObject *, double, double>(args("pos", "time_pace", "time_begin")))
        .def(init<PyObject *, double>(args("pos", "time_pace")))
        .def("at", &Signal::at, args("index"))
        //.def(init<const vector_3 &, double, double>(args("pos", "time_pace", "time_begin")))
        //.def(init<const vector_3 &, double>(args("pos", "time_pace")))
        .def("save", &Signal::save, args("file"))
		.def("load", &Signal::load, args("filename"))
        .staticmethod("load")
        .def("mean_excess_delay", &Signal::meanExcessDelay)
        .def("RMS_delay_spread", &Signal::RMS_DelaySpread)
        .def("multipath_number_10db", &Signal::multipathNumber_10dB)
		.def("trim", &Signal::trim, args("epsilon"))
		.def("parse_CIR", &Signal::parseCIR, args("pulse", "epsilon"))
        .def("push_value", &Signal::pushValue, args("value"))
        .add_property("time_begin", &Signal::timeBegin)
        .add_property("time_end", &Signal::timeEnd)
        .add_property("dt", &Signal::timePace)
        .add_property("length", &Signal::length)
        .add_property("position", &Signal::position)
        .add_property("power", &Signal::power)

        .def("__getitem__", &Signal::at)
        .def("__getslice__", &Signal::subSignal)
        .def("__len__", &Signal::length)
        .def("__str__", &Signal::str)
    ;


    //bool (Fdtd::*addPointSource_Py)(PyObject *, double, double) = &Fdtd::addDipoleSource;
    void (Fdtd::*addReceivePoint_XYZ)(double, double, double) = &Fdtd::addReceivePoint;

    class_<Fdtd>("Fdtd")
        .def(init<>())
        .def("setCoordinateSystem", &Fdtd::setCoordinateSystem, return_self<>(), args("system"))
        .def("setTimePace", &Fdtd::setTimePace, return_self<>(), args("time_pace"))
        .def("Ex", &Fdtd::Ex, args("i", "j", "k"))
        .def("Ey", &Fdtd::Ey, args("i", "j", "k"))
        .def("Ez", &Fdtd::Ez, args("i", "j", "k"))
        .def("Hx", &Fdtd::Hx, args("i", "j", "k"))
        .def("Hy", &Fdtd::Hy, args("i", "j", "k"))
        .def("Hz", &Fdtd::Hz, args("i", "j", "k"))
        .def("Gp", &Fdtd::Gp, return_internal_reference<>(), args("i", "j", "k"))
        .add_property("Sx", &Fdtd::getSizeX)
        .add_property("Sy", &Fdtd::getSizeY)
        .add_property("Sz", &Fdtd::getSizeZ)
        .add_property("S_pml", &Fdtd::getSizePML)
        .add_property("Ds", &Fdtd::getDs)
        .add_property("Dt", &Fdtd::getDt)

        //.def("Gp", &Fdtd::Gp)
        .def("promote", &Fdtd::promote, args("time"))
        .def("addPointSource", &Fdtd::addPointSource_Py, return_self<>(), args("pos", "central_freq", "band_width"))
        .def("getSource", &Fdtd::getPointSource, args("index"))
		//.def("addPointSource", &Fdtd::addPointSource, return_self<>(), args("pos", "central_freq", "band_width"))
        .def("addMedium", &Fdtd::addMedium, return_self<>(), args("shape", "medium"))
        .def("setWall", &Fdtd::setWall, return_self<>(), args("thickness", "medium"))
        //.def("addReceivePoint", &Fdtd::addReceivePoint, return_self<>(), args("pos"))
        .def("addReceivePoint", &Fdtd::addReceivePoint_Py, return_self<>(), args("pos"))
        .def("addReceivePoint", addReceivePoint_XYZ, return_self<>(), args("pos"))
        .def("getReceiveSignal", &Fdtd::getReceiveSignal, args("index"))
        .def("release", &Fdtd::release)
    ;

}

#endif