import java.util.ArrayList;
import java.util.Collections;

/**
 * Solves the unbounded knapsack problem with an algorithm by Lawler as
 * described in "Fast Approximation Algorithms for Knapsack Problems"
 * (1979)
 * 
 *
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class Knapsack{

    /**
     * Creates a new Knapsack instance
     *
     */
    public Knapsack(){}
    
    /**
     * Returns an item with maximal profit/size ratio and minimal size
     *
     * @param xs the items to choose from
     * @return an item out of xs with maximal ratio if xs is not empty;
     *         <code>null</code> else
     */
    public KPItem maxRatio(ArrayList<KPItem> xs){
        if (xs.isEmpty()){
            return null;
        }
        double max = 0;
        double mins = xs.get(0).size;
        KPItem maxit = xs.get(0);
        for (KPItem i: xs){
            double temp = i.profit/i.size;
            if (temp > max){
                max = temp;
                maxit = i;
            }
            else if(temp == max){
                if(i.size < mins){
                    mins = i.size;
                    maxit = i;
                }
            }
        }
        return maxit;
    }

    /**
     * Returns an item with maximal profit
     *
     * @param xs a list of items
     * @return an item out of xs with maximal profit
     */
    public KPItem maxProfit(ArrayList<KPItem> xs){
        double max = 0;
        KPItem maxit = xs.get(0);
        for (KPItem i: xs){
            if (i.profit > max){
                maxit = i;
                max = i.profit;
            }
        }
        return maxit;
    }

    /**
     * Compute an upper bound for the optimum of the unbounded knapsack problem
     *
     * We either use just the most profitable item or put as many copies
     * of the item with the best ratio in it
     *
     * @param xs a list of items
     * @param b the capacity of the knapsack
     * @return <code>-1</code> if the items with the best ratio totally fills the knapsack;
     *         an upper bound for the optimal value otherwise.
     */
    public double upperBound(ArrayList<KPItem> xs, int b){
        KPItem maxr = maxRatio(xs);
        KPItem maxp = maxProfit(xs);
        int ratio = (int)(b/maxr.size);
        if(ratio*maxr.size==b) {
            return (-1);
        }
        return Math.max((ratio*maxr.profit),maxp.profit);
    }
    
    /**
     * Splits the items into a constant number of different ratio classes
     *
     * Two items i,j are in the same class if t*2^l <= i.profit <=
     * t*2^(l+1) and t*2^l <= j.profit <= t*2^(l+1) and
     * floor(i.profit/k*2^l)*2^l == floor(j.profit/k*2^l)*2^l
     *
     * @param large a list of items
     * @param t a parameter to split the items
     * @param p an upper bound for the profit
     * @param k another parameter to split the items
     * @return a list of lists of items which belong to the same ratio class
     */
    public ArrayList<ArrayList<KPItem>> computeQ(ArrayList<KPItem> large,double t, double p, double k){
        ArrayList<ArrayList<KPItem>> q = new ArrayList<>();
        int up = (int)Math.ceil(log2(p/t))+1;
        

        // this is necessary due to Java
        for (int i = 0; i < up;i++){
            q.add(new ArrayList<KPItem>());
        }

        for (KPItem i: large){
            
            // compute the index of the item
            int ind = (int)Math.floor(log2(i.profit/t));
            q.get(ind).add(i);
        }
        return q;
    }
    
    /**
     * Reduces the ratio classes by choosing one item with maximal profit
     *
     * @param q the ratio classes produced by computeQ
     * @return a list of the best items out of every ratio class
     */
    public ArrayList<KPItem> reduceLargeItems(ArrayList<ArrayList<KPItem>> q){
        ArrayList<KPItem> res = new ArrayList<>();
        for (ArrayList<KPItem> qs: q){
            if(qs.size()>0){
                double max = qs.get(0).profit;
                KPItem minit = qs.get(0);
                for(KPItem i: qs){
                    if (i.profit > max){
                        minit = i;
                        max = i.profit;
                    }
                }
                res.add(minit);
            }
        }
        return res;
    }
    
    /**
     * Creates a list, which contains copies of items
     *
     * For each item (p,a), we create items (p*2^l,a*2^l) until a*2^l > b
     * @param xs a list of items
     * @param epsilon the given precision
     * @param b the capacity of the knapsack
     * @return the list of copied items
     */
    public ArrayList<KPItem> copyItems(ArrayList<KPItem> xs, double epsilon, int b){
        ArrayList<KPItem> res = new ArrayList<>();
        
        int k = (int)Math.ceil(log2(4.0/epsilon));


        for (KPItem i: xs){
            res.add(i);
            int p = 2;
            for (int j = 0; j < k; j++){
                if (i.size*p <= b){
                    res.add(new KPItem(i,p));
                }
                p = p*2;
            }
        }
        return res;
    }

    /**
     * Solve the unbounded knapsack problem via the algorithm of lawler
     *
     * @param xs a list of items
     * @param epsilon the precision to solve the problem
     * @param b the capacity of the knapsack
     * @return a {@link Pattern} containing the used items with 
     *         value >= (1-epsilon)OPT
     */
    public Pattern<KPItem> lawler(ArrayList<KPItem> xs,double epsilon, int b){
        
        // compute an upper bound
        double p = upperBound(xs,b);

        // we found the optimal solution
        if (p==-1){
            KPItem max = maxRatio(xs);
            int ratio = (int)(b/max.size);            
            Pattern<KPItem> res = new Pattern<>();
            res.put(max,ratio);
            return res;
        }
        else{
            double t = (epsilon/2)*p;
            double k = (epsilon/2)*t;
            
            // separate the large from the small items
            ArrayList<KPItem> small = new ArrayList<>();
            ArrayList<KPItem> large = new ArrayList<>();
            for(KPItem i: xs){
                if (i.profit <= t){
                    small.add(i);
                }
                else{
                    large.add(i);
                }
            }

            // reduce the problem and solve it
            ArrayList<ArrayList<KPItem>> q = computeQ(large,t,p,k);
            ArrayList<KPItem> redlarge = reduceLargeItems(q);
            ArrayList<KPItem> multredlarge = copyItems(redlarge,epsilon,b);
            ArrayList<Store> pairs = producePairs(multredlarge,b);


            // find the best subset of items
            Store max;
            int phi = 0;
            KPItem maxsmall = null;
            
            // there are no small items
            if (small.isEmpty()){
                max = pairs.get(0);
                double profit = 0;
                for (Store s: pairs){
                    if(s.p > profit){
                        max = s;
                        profit = s.p;
                    }
                }
            }
            else{
                maxsmall = maxRatio(small);
            
                max = pairs.get(0);
                phi = phi(b,maxsmall,max);
                double profit = max.p+phi*maxsmall.profit;
                for (Store s: pairs){
                    int temp = phi(b,maxsmall,s);
                    if (s.p+temp*maxsmall.profit > profit){
                        max = s;
                        phi = temp;
                        profit = s.p+temp*maxsmall.profit;
                    }
                }
            }
            
        
            Pattern<KPItem> res = new Pattern<>();
            
            // compute the items belonging to the best subset
            for(KPItem i : backtrack(max)){
                res.add(i,i.multiplicity);
            }

            // add the best small item
            if(!small.isEmpty()){
                res.add(maxsmall,phi);
            }

            String wit = System.getenv("WITNESS");
            if (wit != null && wit.equals("true")){
                KnapsackWitness kw = new KnapsackWitness(xs,b);
                if (!kw.check(res,epsilon,b)){
                    System.out.println("Could not witness");
                    System.exit(1);
                }
            }
            return res;
        }
    }

    /**
     * Produce all subset of items which could lead to an optimal solution
     * 
     * @param is a list of items
     * @param b the capacity of the knapsack
     * @return all subsets which could lead to an optimal solution
     *         stored in {@link Store}s
     */
    public ArrayList<Store> producePairs(ArrayList<KPItem> is, int b){
        ArrayList<Store> list = new ArrayList<>();
        ArrayList<Store> nlist = new ArrayList<>();
        list.add(new Store(0,0));
        
        // create new candidates
        for (KPItem i: is){
            double pj = i.profit*i.multiplicity;
            double aj = i.size*i.multiplicity;
            for(Store s: list){
                if (s.a+aj <= b){
                    nlist.add(new Store(s,pj,aj,i));
                }
            }

            // eliminate dominated pairs

            list = merge(list,nlist);
            nlist.clear();
        }        

        return list;
    }

    /**
     * Compute how often an item can be added to a knapsack already filled
     *
     * @param b the capacity of the knapsack
     * @param max the item to be added
     * @param s the subset already in the knapsack
     * @return the number of times the item fits into the knapsack
     */
    public int phi(int b, KPItem max, Store s){
        int d = (int)(b-s.a);
        return (int)(d/max.size);
    }

    /**
     * Merge the list of {@link Store}s
     *
     * @param xs the first list
     * @param ys the second list
     * @return all subsets out of both lists which are not dominated by another one
     */
    public ArrayList<Store> merge(ArrayList<Store> xs, ArrayList<Store> ys){
        if (ys.isEmpty()){
            return xs;
        }
        ArrayList<Store> res = new ArrayList<>();
        int l = 0;
        int r = 0;
        double prevp = Math.min(xs.get(0).p,ys.get(0).p);
        
        // iterate through the lists an skip dominated subsets
        // this is possible by the ordering of the subsets
        while(l < xs.size() && r < ys.size()){
            Store left = xs.get(l);
            Store right = ys.get(r);
            if(left.a <= right.a){
                if(left.p >= prevp){
                    res.add(left);
                    prevp = left.p;
                }
                l++;
            }
            else{
                if(right.p >= prevp){
                    res.add(right);
                    prevp = right.p;
                }
                r++;
            }
        }
        // all right items are packed
        while (l < xs.size()){
            Store left = xs.get(l);
            if(left.p >= prevp){
                res.add(left);
            }
            l++;

        }
        // all left items are packed
        while(r < ys.size()){
            Store right = ys.get(r);
            if(right.p >= prevp){
                res.add(right);
            }
            r++;
        }
        return res;
    }
    
    /**
     * Compute all items which belong to a {@link Store}
     * 
     * @param opt the store for which to extract the items
     * @return a list of items belonging to the store
     */
    public ArrayList<KPItem> backtrack(Store opt){
        StoreTree act = opt.tree;
        ArrayList<KPItem> res = new ArrayList<>();
        while(act != null){
            res.add(act.item);
            act = act.next;
        }
        return res;
    }

    /**
     * Computes the logarithm to the base of 2
     * 
     * @param num the argument of the logarithm
     * @return log_2(num)
     */
    public double log2(double num){
        return Math.log(num)/Math.log(2);
    }
}








