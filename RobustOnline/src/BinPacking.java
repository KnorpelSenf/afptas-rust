import java.util.ArrayList;

/**
 * Solve a bin packing instance using MaxMin Resource Sharing
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class BinPacking{
    
    /**
     * The solution to the rounded instance
     */
    Vector<Pattern<BPItem>> rounded;

    /**
     * Creates a new BinPacking instance
     * 
     */
    public BinPacking(){}
    
    
    /**
     * Solves the bin packing problem up to a given precision
     * using a rounding function
     *
     * @param xs a list of items
     * @param epsilon the precision to be reached
     * @param r the {@link RoundingFunction} to be used
     * @return a Pair containing the integral and the fractional solution
     */
    public Pair solve(ArrayList<BPItem> xs, double epsilon, RoundingFunction r){
        // split the items
        ArrayList<BPItem> small = new ArrayList<>();
        ArrayList<BPItem> large = new ArrayList<>();
        for(BPItem i: xs){
            if (i.size <= epsilon/2){
                small.add(i);   
            }
            else{
                large.add(i);
            }
        }
        
        // round the items
        r.initialize(epsilon);
        ArrayList<BPItem> roundedLarge = r.round(large);
        Vector<Pattern<BPItem>> solution = new Vector<>();



        if(!roundedLarge.isEmpty()){
            // solve the rounded instance
            FBinPacking fbp = new FBinPacking();
            Vector<Pattern<BPItem>> roundedSolution = fbp.solve(roundedLarge,epsilon);
            rounded = roundedSolution;

        
            // construct a valid solution 
            solution = r.convert(roundedSolution);      
        }
        
        // round up the solution
        IVector<Pattern<BPItem>> integerSolution = new IVector<>();
        for(Pattern<BPItem> p: solution){
            integerSolution.put(p,(int)Math.ceil(solution.get(p)));
        }
        
        // add the small items via {@link FirstFit}
        IVector<Pattern<BPItem>> res = new FirstFit().firstFit(integerSolution,small);

        String wit = System.getenv("WITNESS");
        if (wit != null && wit.equals("true")){
            BinPackingWitness bpw = new BinPackingWitness(xs);
            if (!bpw.check(res,epsilon,r.getError())){
                System.out.println("Could not witness");
                System.exit(1);
            }
        }
        
        Pair p =new Pair(solution,res);
            
            return  p;
    }
        
}








