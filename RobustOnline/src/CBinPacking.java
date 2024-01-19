import ilog.concert.*;
import ilog.cplex.*;
import java.util.ArrayList;

/**
 * Solves the bin packing problem optimally by using a polynially sized
 * integer program
 *
 *
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class CBinPacking{

    /**
     * Creates a new CBinPacking instance with precision epsilon
     *
     */
    public CBinPacking(){}

    /**
     * Solve the bin packing problem optimally via an polynomially sized
     * integer program
     *
     * @param items the items which describe the problem
     * @return the minimal number of bins needed to pack all items
     */
    public int solve(ArrayList<BPItem> items){        
        try{
            // number of item sizes
            int m = items.size();
            
            double[] size = new double[m];
            int[] freq = new int[m];
            int[] k = new int[m];
            int maxk = 0;
            
            //number of items
            int n = 0;
            int ksum = 0;
            
            

            // initialize the variables
            for (int j = 0; j < m; j++){
                size[j] = items.get(j).size;
                freq[j] = (int)items.get(j).multiplicity;
                n += freq[j];
                k[j] = (int) Math.ceil(log2(1/size[j]))+1;
                if (k[j] > maxk){
                    maxk = k[j];
                }
                ksum += k[j];
            }
            

            // subpattern used to construct pattern
            // u[i][j][i']=0 for i != i'
            int[][][] u = new int[m][maxk][m];
            for (int i=0; i < m; i++){
                for (int j=0; j < k[i];j++){
                    u[i][j][i]=(int)Math.pow(2,j);
                }
            }

            IloCplex cplex = new IloCplex();
            // do not print
            cplex.setOut(null);

            // initialize the lp variables
            IloIntVar[] lambda = cplex.intVarArray(n, 0, n);
            IloIntVar[] x = cplex.intVarArray(n*ksum, 0, n);
            IloIntVar[] mu = cplex.intVarArray(n*ksum, 0, 1);
            
            // the objective value:
            // min \sum_{k} lambda[k]
            IloLinearNumExpr obje = cplex.linearNumExpr();
            
            for (int i = 0; i < n; i++){
                obje.addTerm(1,lambda[i]);
            }
            
            IloObjective obj = cplex.minimize(obje);
            cplex.add(obj);

            // \sum_{k} \sum_{i} \sum_{j} x[k,i,j]u[i,j] = (n_1,...,n_m)
            for(int z = 0; z < m; z++){
                IloLinearNumExpr expr = cplex.linearNumExpr();
                int c = 0;

                for (int kc = 0; kc < n; kc++){
                    for (int i=0; i < m; i++){
                        for (int j=0; j <k[i]; j++){
                            expr.addTerm(u[i][j][z],x[c]);
                            c++;
                        }
                    }
                }
                cplex.addEq(expr,freq[z]);
            }

            // (\sum_{i} \sum_{j} mu[k,i,j]u[i,j]) (s_1,...,s_m) <= 1
            int c = 0;
            for(int ck = 0; ck < n; ck++){
                IloLinearNumExpr expr = cplex.linearNumExpr();
                for(int i = 0; i < m; i++){
                    for(int j = 0; j < k[i]; j++){
                        for(int h=0; h <m; h++){
                            expr.addTerm(u[i][j][h]*size[h],mu[c]);
                        }
                        c++;
                    }
                }
                cplex.addLe(expr,1);
            }

            // x[k,i,j] <= n*mu[k,i,j]
            for(int i = 0; i < n*ksum; i++){
                IloLinearNumExpr expr1 = cplex.linearNumExpr();
                expr1.addTerm(1,x[i]);

                IloLinearNumExpr expr2 = cplex.linearNumExpr();
                expr2.addTerm(n,mu[i]);
                cplex.addLe(expr1,expr2);
            }
            
            // lambda[k] >= x[k,i,j]
            for(int kc = 0; kc < n; kc++){
                for (int j = kc*ksum; j < (kc+1)*ksum; j++){
                    IloLinearNumExpr expr1 = cplex.linearNumExpr();
                    expr1.addTerm(1,lambda[kc]);

                    IloLinearNumExpr expr2 = cplex.linearNumExpr();
                    expr2.addTerm(1,x[j]);
                    cplex.addGe(expr1,expr2);
                }
            }
         
            // solve the problem
            cplex.solve();
            return (int) cplex.getObjValue();
        }
        catch (Exception e) {
            for (BPItem i: items){
                System.out.println(i.id);
                System.out.println(i.size);
            }

            System.err.println("Concert exception caught: " + e);
        } 
        return 0;
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

    





