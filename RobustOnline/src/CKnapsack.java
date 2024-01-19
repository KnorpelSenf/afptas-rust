import ilog.concert.*;
import ilog.cplex.*;
import java.util.ArrayList;

/**
 * Solve the unbounded knapsack problem via CPLEX
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class CKnapsack{

    /**
     * Creates a new CKNapsack instance
     *
     */
    public CKnapsack(){}
    
    /**
     * Solves the unbounded knapsack problem
     * 
     * @param xs a list of items with profit and size
     * @param b the capacity of the knapsack
     * @return a pattern containing the optimal solution
     */
    public Pattern<KPItem> solve(ArrayList<KPItem> xs,int b){
        
        try {
            IloCplex cplex = new IloCplex();
            // do not print
            cplex.setOut(null);
            
            //create an array of xs.size() integer variables between 0
            //and Integer.MAX_VALUE
            IloIntVar[] x = cplex.intVarArray(xs.size(), 0, Integer.MAX_VALUE);
            
            // create the objective function
            IloLinearNumExpr expr = cplex.linearNumExpr();
            
            // create the constraints
            IloLinearNumExpr bound = cplex.linearNumExpr();
            for(int i = 0; i < xs.size(); i++){
                KPItem it = xs.get(i);
                
                // add the profit to the objective function
                expr.addTerm(it.profit,x[i]);
                
                // add the size to the constraints
                bound.addTerm(it.size,x[i]);
            }            
            
            // maximize the objective function
            IloObjective obj = cplex.maximize(expr);
            
            // add the objective functiont to the model
            cplex.add(obj);
            
            // add the constraints to the model
            cplex.addLe(bound,b);
            

            
            // solve the model
            cplex.solve();
            
            // extract the solution
            double[] s = cplex.getValues(x);
            

            Pattern<KPItem> res = new Pattern<>();
            for (int i = 0; i <xs.size() ; i++){
                res.add(xs.get(i),(int)s[i]);
            }
            return res;
        } catch (Exception e) {
            System.err.println("Concert exception caught: " + e);
        }   
        return null;
    }
}




