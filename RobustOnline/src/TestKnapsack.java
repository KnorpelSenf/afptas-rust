import java.util.ArrayList;
import java.util.Random;
import java.io.File;
import java.io.FileWriter;
import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

/**
 * Runs several tests on the Knapsack and the CKnapsack algorithms
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class TestKnapsack{

    /**
     * Upper bound of the weight -1
     */
    public static final int UPPER_WEIGHT = 99;

    /**
     * Upper bound of the profit -1
     */
    public static final int UPPER_PROFIT = 99;

    /**
     * Ratios used to compute the capacity
     */
    public static final double[] ratios = {0.75,0.5,0.25};
    
    /**
     * Approximation factors
     */
    public static final double[] approx = {0.1,0.09,0.08,0.07,0.06,0.05};

    /**
     * Test 5000 instances for each ratio/approx combination
     * 
     * @param args ignored
     */
    public static void main(String[] args){
        // find a unique file name for the test 
        DateFormat df = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        Date d = new Date();
        String date = df.format(d);

        try{
            // initialize all variables at one time to save memory
            Random gen = new Random();
            ArrayList<KPItem> xs = new ArrayList<>();
            int size;
            int capacity;
            Knapsack kp = new Knapsack();
            CKnapsack cp = new CKnapsack();
            double before;
            double after;
            int weight;
            int profit;
            double timeapprox;
            double valueapprox;
            double timeopt;
            double valueopt;

            for(double epsilon: approx){
                for (double r : ratios){
                    // open the file an write the first line
                    File file = new File("../results/"+date+r+epsilon);
                    file.createNewFile();
                    FileWriter writer = new FileWriter(file);
                    writer.write("amount timeapprox valueapprox timeopt valueopt theory\n");

                    for(int i = 1000; i < 6001; i++){
                        size = 0;
                        
                        // create the instance
                        for (int j = 0; j < i; j++){
                            // prevent weight and profit from having value 0
                            weight = gen.nextInt(UPPER_WEIGHT)+1;
                            profit = gen.nextInt(UPPER_PROFIT)+1;
                            xs.add(new KPItem(""+j, profit, weight));
                            size = size + weight;
                        }
                        
                        capacity = (int) (size*r);
                        
                        // solve the problem approximately
                        before = System.currentTimeMillis();
                        Pattern<KPItem> res = kp.lawler(xs,epsilon,capacity);
                        after = System.currentTimeMillis();
                        timeapprox = after-before;
                        
                        // get the value of the approximate solution
                        valueapprox = 0;
                        for (KPItem it: res){
                            valueapprox = valueapprox + res.get(it);   
                        }
                        if(valueapprox <= 0){
                            valueapprox = Integer.MAX_VALUE;
                        }
                       
                        // solve the problem optimally
                        before = System.currentTimeMillis();
                        res = cp.solve(xs,capacity);
                        after = System.currentTimeMillis();
                        timeopt = after-before;
                        
                        // get the value of the optimal solution
                        valueopt = 0;
                        for (KPItem it: res){
                            valueopt = valueopt + res.get(it);
                        }
                        if(valueopt <= 0){
                            valueopt = Integer.MAX_VALUE;
                        }

                        // write the results
                        writer.write(i+" "+timeapprox+" "+valueapprox+" "+timeopt+" "+valueopt+" "+((1-epsilon)*valueopt)+"\n");
                        xs.clear();
                    }
                    writer.close();
                }
            }
        }
        catch(Exception e){
            System.out.println(e);
        }
    }
}
