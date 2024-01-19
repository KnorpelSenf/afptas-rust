import java.util.ArrayList;
import java.util.HashMap;
/**
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class OnlineBinPacking{

    /**
     * The used robust rounding function
     */
    RobustRoundingFunction round;
    
    /**
     * The way to get the next item
     */
    Input input;
    
    /**
     * Stores the time used to produce the last solution
     */
    double timeapprox;
    
    /**
     * Stores the value of the last solution
     */
    double valueapprox;
    

    /**
     * The migration factor of the last solution
     */
    double migrationFactor;

    /**
     * A counter to describe the number of items treated
     */
    int counter;

    /**
     * Creates a new instance of OnlineBinPacking
     *
     * @param round the rounding function to be used
     * @param input the way to get the next item
     */
    public OnlineBinPacking(RobustRoundingFunction round, Input input){
        this.round = round;
        this.input = input;
        counter = 0;
        System.out.println("counter migrationFactor timeApprox valueApprox");
    }


    /**
     * Starts the robust online algorithm
     *
     * @param epsilon the approximation factor
     */
    public void solve(double epsilon){
        
        double before;
        double after;
        round.initialize(epsilon);
        double largeSize = 0;
        ArrayList<BPItem> large = new ArrayList<>();
        ArrayList<BPItem> small = new ArrayList<>();
        BPItem next = input.getNextItem();
        
        counter++;
        BinPacking bp = new BinPacking();        
        FirstFit ff = new FirstFit();
        RoundingFunction simple = new OldRobustRoundingFunction();
        if (next.size > epsilon/2){
            large.add(next);
            largeSize += next.size;
        }
        else{
            small.add(next);
        }
        // solve the problem optimally until the threshold is reached
        
        while (largeSize <= round.getThreshold()){

            before = System.currentTimeMillis();
            Pair pxy = bp.solve(large,epsilon/2,simple);
            
            after = System.currentTimeMillis();
            timeapprox = after-before;

            IVector<Pattern<BPItem>> res = ff.firstFit(pxy.integral,small);
       
            valueapprox = res.sum();
            migrationFactor = 0;
            for (BPItem it: large){
                migrationFactor = migrationFactor + it.size;
            }
            for(BPItem it: small){
                migrationFactor = migrationFactor + it.size;
            }
            next = input.getNextItem();
            counter++;
            if(next == null){
                return;
            }

            show();
            if (next.size > epsilon/2){
                large.add(next);
                largeSize += next.size;
            }
            else{
                small.add(next);
            }
            
        }
        // we finished the first phase
        System.out.println("Finishes first phase");

        // start the next phase as we have enough total size
        before = System.currentTimeMillis();
        bp.solve(large,epsilon/2,round);
        
        // construct the initial pair
        Vector<Pattern<BPItem>> x = bp.rounded;
        IVector<Pattern<BPItem>> y = new IVector<>();
        for(Pattern<BPItem> p: x){
            y.put(p,(int)Math.ceil(x.get(p)));
        }
        Pair pxy = new Pair(x,y);

        // the non rounded integral solution
        IVector<Pattern<BPItem>> yw = round.convert(pxy.integral);

        // the complete solution
        IVector<Pattern<BPItem>> ywithsmall = ff.firstFit(yw,small);   
        after = System.currentTimeMillis();
        timeapprox = after-before;
        valueapprox = ywithsmall.sum();
        
        // compute the migration Factor
        migrationFactor = 0;
        for (BPItem it: large){
            migrationFactor = migrationFactor + it.size;
        }
        for(BPItem it: small){
            migrationFactor = migrationFactor + it.size;
        }
        show();
        HashMap<Pattern<BPItem>,Pattern<BPItem>> belongsToInvers = ff.belongsToInvers;
        
        while(true){
            before = System.currentTimeMillis();

            // improve the solution!
            Improve imp = new Improve();
            
            // pack the new item
            boolean isSmall= true;
            next = input.getNextItem();
            counter++;
            if(next == null){
                return;
            }
            if (next.size > epsilon/2){
                isSmall = false;
            }
            migrationFactor = 0;
            // the new item is large and needs to be integrated in the
            // rounding
            if(!isSmall){
                pxy = imp.improve(large,pxy,round.improve(),epsilon);
                large.add(next);

                round.add(next,pxy);
                // update the rounding function if necessary 
                round.update(pxy);

                // construct the next full solution by comparing the
                // integral solutions
                IVector<Pattern<BPItem>> ynext = round.convert(pxy.integral);

                ArrayList<BPItem> smallToRepack = new ArrayList<>();
                ArrayList<Pattern<BPItem>> toReplace = new ArrayList<>();

                for (Pattern<BPItem> p: yw){
                    int tmp = 0;
                    if (ynext.get(p) != null){
                        tmp = ynext.get(p);
                    }
                    int val = 0;
                    Integer vali = yw.get(p);
                    if (vali != null){
                        val = vali;
                    }

                    // we removed a pattern. Get the small items out of it!
                    if (val-tmp == 1){
                        Pattern<BPItem> filled = belongsToInvers.get(p);
                        if(filled != null){
                            for (BPItem i: filled){

                                if (i.size <= epsilon/2){
                                    smallToRepack.add(i);
                                    migrationFactor = migrationFactor+i.size;
                                }
                            }
                            if (!filled.equals(p)){
                                belongsToInvers.remove(p);
                            }
                        }
                    }
                    if(val-tmp == 0){
                        toReplace.add(p);
                    }
                }

                // which pattern are new
                ArrayList<Pattern<BPItem>> toAdd = new ArrayList<>();
                for (Pattern<BPItem> p: ynext){
                    if (yw.get(p) == null){
                        toAdd.add(p);
                        for (BPItem it: p){
                            migrationFactor = migrationFactor + it.size;                        
                        }

                    }
                }
                migrationFactor = migrationFactor / next.size;

                // create the new solution
                IVector<Pattern<BPItem>> ynexttmp = new IVector<Pattern<BPItem>>();
                for (Pattern<BPItem> p: toAdd){
                    ynexttmp.put(p,1);
                }

                for(Pattern<BPItem> p: toReplace){
                    if(belongsToInvers.get(p) != null){
                        ynexttmp.put(belongsToInvers.get(p),1);
                    }
                    else{
                        ynexttmp.put(p,1);
                    }
                }

                ywithsmall = ff.firstFit(ynexttmp,smallToRepack);
                yw = ynext;

            }
            // the new item is small and the rounding does not need to
            // change
            else{
                ArrayList<BPItem> smalls = new ArrayList<>();
                smalls.add(next);
                ywithsmall = ff.firstFit(ywithsmall,smalls);
                migrationFactor = next.size;
            }

            belongsToInvers.putAll(ff.belongsToInvers);

            after = System.currentTimeMillis();
            timeapprox = after-before;
            valueapprox = ywithsmall.sum();
            show();
        }
    }

    /**
     * Displays the current solution
     *
     */
    public void show(){
        System.out.println(counter+" "+migrationFactor+" "+timeapprox+" "+valueapprox);
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
