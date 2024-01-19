import java.util.ArrayList;
import java.util.HashMap;
import java.util.Comparator;
import java.util.Collections;

/**
 * This improves a pair of bin packing solutions
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class Improve{

    /**
     * Creates a new instance of Improve
     * 
     */
    public Improve(){}
    
    /**
     * Improves a pair of bin packing solutions
     *
     * @param items the instance of the problem
     * @param pair the pair consisting of the solutions
     * @param alpha the value to improve to
     * @param delta the approximation ratio to use
     * @return a pair of solutions which are improved by alpha
     */
    public Pair improve(ArrayList<BPItem> items, Pair pair, int alpha, double delta){
        Vector<Pattern<BPItem>> x = pair.fractional;
        IVector<Pattern<BPItem>> y = pair.integral;
        
        // split x into a fixed an a variable part
        double val = (2*alpha*(1.0/delta +1))/x.sum();
        
        Vector<Pattern<BPItem>> xv = x.scale(val);
        Vector<Pattern<BPItem>> xf = x.scale(1-val);

        // compute the right hand side
        HashMap<BPItem,Double> freq = new HashMap<>();

        for (Pattern<BPItem> p: x){
            for (BPItem i: p){
                Double tmp = freq.get(i);
                if(tmp == null){
                    tmp = 0.0;
                }
                freq.put(i,tmp+(x.get(p)*p.get(i)));
            }
        }

        // compute A(xf)
        for (Pattern<BPItem> p: x){
            for (BPItem i: p){
                Double tmp = freq.get(i);
                if (tmp == null){
                    tmp = 0.0;
                }
                freq.put(i,tmp-(xf.get(p)*p.get(i)));
            }
        }

        // create the instance corresponding to freq
        ArrayList<BPItem> redInstance = new ArrayList<>();

        for (BPItem i: freq.keySet()){
            if (freq.get(i) > 0){
                redInstance.add(new BPItem(i.id, i.size, freq.get(i)));
            }
        }

        // solve the problem approximately 
        FBinPacking fbp = new FBinPacking();
        Vector<Pattern<BPItem>> xprime = fbp.solve(redInstance,delta/2);
        
        Vector<Pattern<BPItem>> xres = new Vector<>();
        IVector<Pattern<BPItem>> yprime = new IVector<>();
        
        // test whether the solution is good enough already
        if (xf.add(xprime).sum() >= x.sum()){
            xres = x;
            yprime = y;
        }
        else{
            // improve the solution

            // sort y in order to have the smallest components available
            ArrayList<Pattern<BPItem>> ys = new ArrayList<>();
            for (Pattern<BPItem> p: y){
                ys.add(p);
            }

            final IVector<Pattern<BPItem>> yfinal = y;
            
            Collections.sort(ys,new Comparator<Pattern<BPItem>>(){
                    public int compare(Pattern<BPItem> p1, Pattern<BPItem> p2){
                        return Double.compare(yfinal.get(p1),yfinal.get(p2));
                    }});
            
            // compute l such that \sum_{i=1}^{l} y[i] <= (m+2)(1/e +2)
            ArrayList<Pattern<BPItem>> smallest = new ArrayList<>();
            double smallestSum = 0;
            int l = 0;
            do{
                smallest.add(ys.get(l));
                smallestSum = smallestSum + y.get(ys.get(l));
                l++;
            }
            while(smallestSum <= (items.size())/(1.0/delta +2));
            
            // we go one step too far as smallestSum is now too big
            l--;

            // get the upper half of x
            Vector<Pattern<BPItem>> smallestX = new Vector<>();
            for (int i = 0; i < l; i++){
                Pattern<BPItem> p = ys.get(i);
                if (p != null){
                    smallestX.put(p,x.get(p));
                }
            }

            // reduce the components of xbar
            Vector<Pattern<BPItem>> xbar = xprime.add(smallestX);
            ReduceComponents rd = new ReduceComponents();
            
            xbar = rd.reduce(xbar,items);
            
            Vector<Pattern<BPItem>> xnew = new Vector<>();
            IVector<Pattern<BPItem>> ynew = new IVector<>();

            // compute the lower half of x and y
            for (int i = l; i <  ys.size(); i++){
                Pattern<BPItem> p = ys.get(i);
                Double tmp = xf.get(p);
                if (tmp == null){
                    tmp = 0.0;
                }
                xnew.put(p,tmp);
                ynew.put(p,y.get(p));
            }

            xres = xnew.add(xbar);

            // take the better rounded up values
            for (Pattern<BPItem> p: xres){
                int tmp = (int)Math.ceil(xres.get(p));
                if (ynew.get(p) != null){
                    int tmp2 = ynew.get(p);
                    tmp = Math.max(tmp,tmp2);
                }
                yprime.put(p,tmp);
            }
        }

        // find a vector d with d[i] <= yprime[i]-xres[i]        
        Vector<Pattern<BPItem>> nxres = xres.scale(-1);
        Vector<Pattern<BPItem>> diff = yprime.toVector().add(nxres);

        IVector<Pattern<BPItem>> d = new IVector<>();
        int dsum = 0;
        double upperBound = 2*alpha*(1.0/delta +2)+items.size()+1;
        
        for (Pattern<BPItem> p: diff){
            double tmp = diff.get(p);
            int tmpint = (int)Math.floor(tmp);
            if (dsum + tmpint <= upperBound){
                d.put(p,-tmpint);
                dsum = dsum + tmpint;
            }
            else{
                d.put(p,-(int)Math.floor(upperBound-dsum));
                break;
            }
        }
        
        // subtract d from yprime
        IVector<Pattern<BPItem>> yres = yprime.add(d);
        return new Pair(xres,yres);
    }
}
