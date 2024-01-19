import java.util.ArrayList;
import java.io.File;
import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.io.PrintStream;
import java.io.FileOutputStream;

/**
 * Runs several tests on the online bin packing algorithm
 * @author Sebastian Berndt <sbe@informatik.uni-kiel.de>
 */

public class TestBinPacking{
    
    public static void main(String[] args){
        // the instances are stored in TestBinPacking.xml
        int n = Integer.parseInt(args[1]);
        ParseInput p = new ParseInput(args[0], 1, n);
        double [][] sizes = p.sizes;
        double epsilon = Double.parseDouble(args[2]);

        DateFormat df = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        Date d = new Date();
        String date = df.format(d);
        try{
            // try instance with given epsilon and the instance stored in sizes
            PrintStream io = System.out;
            
                

                RobustRoundingFunction nround = new NewRobustRoundingFunction();
                Input input = new TestInput(n-1,sizes[0]);
                OnlineBinPacking obp = new OnlineBinPacking(nround,input);
                obp.solve(epsilon);
                io.println("Finished NEW");

                input = new TestInput(n-1,sizes[0]);
                RobustRoundingFunction oround = new OldRobustRoundingFunction();
                obp = new OnlineBinPacking(oround,input);
                obp.solve(epsilon);
                io.println("Finished OLD");

            }
        
        catch(Exception e){
            System.out.println(e);
        }
    }
}



