import java.util.ArrayList;

/**
 * Checks wether a MaxMin solution is good enough
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class MaxMinWitness<I,T>{

    /**
     * Creates a new MaxMinWitness instance
     */
    public MaxMinWitness(){
        
    }

    /**
     * Tests whether a solution to the MaxMin problem is within a given
     * distance and feasible
     * 
     * @param price the final price vector
     * @param fx the final value of f(x)
     * @param epsilon the desired approximation factor
     * @return whether the solution is feasible and good enough
     */
    boolean check(ArrayList<Double> price, ArrayList<Double> fx, double epsilon){
        double min = Integer.MAX_VALUE;
        for (double fxi: fx){
            if (fxi < min){
                min = fxi;
            }
        }
        double value = 0;
        double sum = 0;
        for (int i = 0; i < price.size(); i++){
            value = value + price.get(i)*fx.get(i);
            sum = sum + price.get(i);
        }
        return (((1+epsilon)/(1-epsilon)*min >= (1-epsilon)*value) && Math.abs(1-sum) <= epsilon);
    }



}
