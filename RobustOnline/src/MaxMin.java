import java.util.ArrayList;
import java.util.Set;

/**
 * Solves the maxmin resource sharing problem with an algorithm by
 * Jansen as described in "An approximation algorithm for the general
 * max-min resource sharing problem"
 *
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 * @param <I> the additional input which is needed
 * @param <T> the entries of the result vector
 */
public class MaxMin<I,T>{
    
    /**
     * The approximative block solver used throughout the algorithm
     */
    ABS<I,T> abs;
    
    /**
     * The additional input
     */
    I input;
    
    /**
     * Creates a new MaxMin instance with a given approximative block
     * solver and input
     * 
     * @param abs the approximative block solver
     * @param input the input
     */
    public MaxMin(ABS<I,T> abs, I input){
        this.abs = abs;
        this.input = input;
    }
    
    /**
     * Computes the first derivative of the theta function
     *
     * @param t the approximation factor
     * @param theta the value for theta
     * @param fx the list containing the results of f(x)
     * @return the value of the first derivative of the theta function
     */
    public double computeThetaF(double t, double theta, ArrayList<Double> fx){
        double sum = 0;
        for (double x: fx){
            sum = sum + theta/(x-theta);
        }
        return sum*(t/fx.size());
    }
    
    /**
     * Compute the optimal theta 
     *
     * @param t the approximation factor
     * @param fx the list containing the results of f(x)
     * @param prec the precision for which to calculate theta
     * @return a theta with computeThetaF(t,theta,fx)=1
     */
    public double findTheta(double t, ArrayList<Double> fx, double prec){
        double upper = fx.get(0);
        for(double x: fx){
            if (x < upper){
                upper = x;
            }
        }
        double lower = 0;
        double act = (upper+lower)/2;

        // use binary search to find theta
        while(true){
            act = (upper+lower)/2;
            double val = computeThetaF(t,act,fx);
            
            if(Math.abs(val-1) < prec){
                break;
            }
            else if((val-1) < 0){
                lower = act;
            }
            else{
                upper = act;
            }
        }
        return act;
    }
    
    /**
     * Computes the price vector to be used by the block solver
     * 
     * @param fx the list containing the results of f(x)
     * @param t the approximation factor
     * @param theta the value given by the potential function
     * @return the price vector
     */
    public ArrayList<Double> computePrice(ArrayList<Double> fx, double t, double theta){
        double r = t/fx.size();
        ArrayList<Double> res = new ArrayList<>();
        for(Double x: fx){
            res.add(r*theta/(x-theta));
        }
        return res;
    }

    /**
     * Multiply two lists of numbers with the same length
     * 
     * @param x the first list
     * @param y the second list
     * @return a list l with l.get(i)==x.get(i)*y.get(i)
     */
    public double multiply(ArrayList<Double> x, ArrayList<Double> y){
        double res = 0;
        for(int i = 0; i < x.size(); i++){
            res = res+x.get(i)*y.get(i);
        }
        return res;
    }
    
    /**
     * Apply the {@link Function}s to a {@link Vector}
     *
     * @param fs the list of {@link Function}s
     * @param x the {@link Vector} given as argument
     * @return a list containing the values of f(x)
     */
    public ArrayList<Double> applyAll(ArrayList<Function<T>> fs, Vector<T> x){
        ArrayList<Double> res = new ArrayList<>();
        for (Function<T> f: fs){
            res.add(f.apply(x));
        }
        return res;
    }
    
    /**
     * Performs a line search in order to compute the step width
     *
     * @param fx a list containing the values of f(x)
     * @param fy a list containing the values of f(y)
     * @param theta the value used in the potential function
     * @param t the precision to be used in the algorithm
     * @param epsilon the approximation factor
     * @return a near-optimal step width
     */
    public double lineSearch(ArrayList<Double> fx, ArrayList<Double> fy, double theta, double t, double epsilon){
        double up = 1.0;
        double low = 0.0;
        
        // perform a binary search
        while (low < (1-epsilon)*up){
            double act = (up+low)/2;
            
            // test if the potential function is still defined
            boolean defined = true;
            for (int i = 0; i < fx.size(); i++){
                if (fx.get(i)+act*(fy.get(i)-fx.get(i)) <= theta){
                    defined = false;
                }
            }
            if (!defined){
                up = act;
            }
            else{
                double val = derivativePot(act,fx,fy,t,theta);
                if (val > 0){
                    low = act;
                }
                else{
                    up = act;
                }
            }
        }
        return (low+up)/2;
    }

    /**
     * Computes the value of the derivative of the simplified potential function
     *
     * @param tau the step width
     * @param fx a list containing the values of f(x)
     * @param fy a list containing the values of f(y)
     * @param t the precision of the algorithm
     * @param theta the values used in the potential function
     * @return the value of the derivative 
     */
    public double derivativePot(double tau, ArrayList<Double> fx, ArrayList<Double> fy,double t,double theta){
        double res = 0;
        for (int i = 0; i < fx.size(); i++){
            res = res + (fy.get(i)-fx.get(i))/(fx.get(i)+tau*(fy.get(i)-fx.get(i))-theta);
        }
        return (res*t/fx.size());
    }

    /**
     * Computes the parameter which indicates when to stop
     * 
     * @param p the price vector
     * @param fx a list containing the values of f(x)
     * @param fy a list continaing the values of f(y)
     * @return the stop parameter
     */
    public double computeV(ArrayList<Double> p, ArrayList<Double> fx, ArrayList<Double> fy){
        double a = multiply(p,fy);
        double b = multiply(p,fx);
        return (a-b)/(b+a);
    }

    /**
     * Compute the step width 
     * This is not used when a line search is feasible
     *
     * @param t the approximation factor
     * @param theta the value used in the potential function
     * @param v the stop parameter
     * @param price the price vector
     * @param fx a list containing the values of f(x)
     * @param fy a list containing the values of f(y)                            
     * @return the step width
     */
    public double computeTau(double t, double theta, double v, ArrayList<Double> price, ArrayList<Double> fx, ArrayList<Double> fy){
        double a = t*theta*v;
        double b = 2*fx.size()*(multiply(price,fx)+multiply(price,fy));
        return a/b;
    }
    
    /**
     * Computes a starting solution for the algorithm
     *
     * @param m the number of functions
     * @return a {@link Vector} containing the starting solution
     */
    public Vector<T> computeStart(int m){
        Vector<T> res = new Vector<>();
        for (int i = 0; i < m; i++){
            // solve ABS(e_m,1/2)
            res = res.add(abs.solve(unit(i,m),0.5,input));
        }
        for (T i: res){
            // scale the solution
            res.put(i,res.get(i)/m);
        }
        return res;
    }

    /**
     * Creates a list l which contains a single 1 and 0 everywhere else
     *
     * @param ind the index of the 1
     * @param m the size of the list
     * @return a list l with l.get(ind)=1 and l.get(i)=0 for all i != ind
     */
    public ArrayList<Double> unit(int ind, int m){
        ArrayList<Double> res = new ArrayList<>();
        for (int i = 0; i < m; i++){
            if (i==ind){
                res.add(1.0);
            }
            else{
                res.add(0.0);
            }
        }
        return res;
    }

    /**
     * Compute a solution to the maxmin resource sharing problem
     * 
     * @param fs the list of (convex) functions
     * @param epsilon the precision
     * @return a {@link Vector} containing a solution with value >= (1-epsilon)OPT
     */
    public Vector<T> maxmin(ArrayList<Function<T>> fs, double epsilon){
        
        Vector<T> x = computeStart(fs.size());
        double actepsilon = 0.25;
        double prec = epsilon*epsilon/fs.size();
        ArrayList<Double> fx = applyAll(fs,x);
        ArrayList<Double> price = null;
        // iterate over the scaling phases
        do{
            actepsilon = actepsilon/2;
            double t = epsilon/6;
            double minx = Double.MAX_VALUE;
            for (double fxi : fx){
                if (fxi < minx){
                    minx = fxi;
                }
            }
            // start the scaling phase
            while(true){
                double theta = findTheta(t,fx,prec);
                price = computePrice(fx,t,theta);
                Vector<T> y = abs.solve(price,t,input);
                ArrayList<Double> fy = applyAll(fs,y);
                double miny = Double.MAX_VALUE;
                for (double fyi: fy){
                    if (fyi < miny){
                        miny = fyi;
                    }
                }
                double v = computeV(price,fx,fy);
                if (v <= t || miny >= minx*(1-actepsilon)/(1-2*actepsilon)){
                    break;
                }
                double tau = lineSearch(fx,fy,theta,t,epsilon);
                ArrayList<Double> fx1 = new ArrayList<>();
                for (int i = 0; i < fx.size(); i++){
                    fx1.add((1-tau)*fx.get(i)+tau*fy.get(i));
                }
                x = x.scale(1-tau).add(y.scale(tau));
                fx = fx1;
            }
        }
        while(actepsilon > epsilon);
        
        String wit = System.getenv("WITNESS");
        if (wit != null && wit.equals("true")){
            MaxMinWitness mmw = new MaxMinWitness();
            if (!mmw.check(price,fx,epsilon)){
                System.out.println("Could not witness");
                System.exit(1);
            }
        }
        return x;
    }
}











