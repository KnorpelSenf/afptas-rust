import org.apache.commons.math3.linear.*;
import java.util.ArrayList;

/**
 * Reduces the number of non-zero components of a bin packing solution
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class ReduceComponents{
    
    /**
     * Creates a new ReduceComponents instance
     * 
     */
    public ReduceComponents(){}



    /**
     * Converts a bin packing instance and a solution to it into the
     * corresponding matrix
     *
     * @param v the solution of the problem
     * @param items the instance
     * @return a matrix corresponding to the columns used in the solution
     */
    public RealMatrix convertToMatrix(Vector<Pattern<BPItem>> v, ArrayList<BPItem> items){
        int n = v.getSize();
        int m = items.size();
        RealMatrix a = new Array2DRowRealMatrix(m,n);
        int i = 0;
        // start the converting
        for (Pattern<BPItem> p: v){
            for(int j=0; j < items.size(); j++){
                BPItem it = items.get(j);
                Integer val = p.get(it);
                if (val != null){
                    a.setEntry(j,i,val);
                }
                else{
                    a.setEntry(j,i,0);
                }
            }
            i++;
        }
        
        return a;
    }
    
    /**
     * Converts a vector describing a bin packing solution to an array
     *
     * @param v the vector containing the solution
     * @return an array describing the solution
     */
    public double[] convertToArray(Vector<Pattern<BPItem>> v){
        double[] res = new double[v.getSize()];
        int i = 0;
        // use the iterator of v to describe the ordering
        for (Pattern<BPItem> p: v){
            res[i] = v.get(p);
            i++;
        }
        return res;
    }
    
    /**
     * Converts an array describing a new bin packing solution and a vector
     * describing a previous solution into a vector describing the new solution
     * 
     * @param v the vector describing the previous solution
     * @param y the array describing the new solution
     * @return a vector describing the new solution
     */
    public Vector<Pattern<BPItem>> convertFromArray(Vector<Pattern<BPItem>> v, double[] y){
        Vector<Pattern<BPItem>> res = new Vector<>();
        int i = 0;
        // use the iterator of v to describe the ordering
        for(Pattern<BPItem> p: v){
            if (y[i] > 0){
                res.put(p,y[i]);
            }
            i++;
        }
        return res;
    }
    
    /**
     * Creates an array of size dimension filled with fill
     * 
     * @param dimension the dimension of the array
     * @param fill the value to fill the array
     * @return an array of the given dimension filled with fill
     */
    public int[] createRow(int dimension, int fill){
        int[] res = new int[dimension];
        for (int i=0; i< dimension; i++){
            res[i] = fill;
        }
        return res;
    }
    
    /**
     * Reduces the number of non-zero entries of a bin packing solution
     * 
     * @param v the bin packing solution
     * @param items the bin packing instance
     * @return a vector containing a solution with less non-zero entries
     * and the same objective value
     */
    public Vector<Pattern<BPItem>> reduce(Vector<Pattern<BPItem>> v, ArrayList<BPItem> items){
        // Do we need to do something
        if(v.getSize() <= items.size()){
            return v;
        }

        // Create the linear equation system
        RealMatrix matrix = convertToMatrix(v,items);
        double[] y = convertToArray(v);
        int n = matrix.getColumnDimension();
        int m = matrix.getRowDimension();
        int[] rows = createRow(m,1);

        // describes which columns are currently active in the equation
        // system
        int[] columns = createRow(n,0);
        for (int i = 0; i < m+1; i++){
            columns[i] = 1;
        }
        
        // reduce all superfluous non-zero entries
        int counter = m+1;
        while (counter < n){
            // get the corresponding submatrix of the matrix
            RealMatrix sub = matrix.getSubMatrix(rows,columns);
            
            // factorize the matrix into its singular vectors
            SingularValueDecomposition svd = new SingularValueDecomposition(sub);
            
            // extract the right singular vectors
            RealMatrix vs = svd.getV();
            
            // the kernel of sub is described by the last columns of vs
            double[] z = vs.getColumn(vs.getColumnDimension()-1);
            
            // find an i such that |y[i]/z[i]| is minimal
            double sigma = (double) Integer.MAX_VALUE;
            int sigmaCol = 0;
            for (int j=0; j < n;j++){
                if(columns[j] == 1){
                    if (Math.abs(y[j]/z[j]) <= Math.abs(sigma)){
                        sigma = - Math.abs(y[j]/z[j]);
                        sigmaCol = j;
                    }
                }
            }
            
            // reduce the components
            for (int j = 0; j < n; j++){
                if (columns[j] == 1){
                    y[j] = y[j]+sigma*z[j];
                }
            }

            // the component corresponding to sigmaCol is reduced to
            // zero
            columns[sigmaCol] = 0;
            columns[counter] = 1;
            counter++;
        }
        
        Vector<Pattern<BPItem>> res = convertFromArray(v,y);
        return res;
    }
}

