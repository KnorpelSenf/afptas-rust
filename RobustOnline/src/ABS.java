import java.util.ArrayList;

/**
 * ABS is the abstract class for the {@link MaxMin} algorithm,
 * representing the approximate block solver, which is used to generate
 * the columns. 
 *
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 * @param <I> the type of the input for the block solver
 * @param <T> the type of the output of the block solver, containted by a {@link Vector}
 */
public interface ABS<I,T>{
    
    /**
     * Finds a vector x in the block with an approximation ratio of (1-t)
     * 
     * @param price the price vector describing the block problem
     * @param t the precision for which to solve the problem
     * @param input the input describing the functions and the block
     * @return a {@link Vector} containing the next column to be
     *         used by the {@link MaxMin} algorithm
     */
    public Vector<T> solve(ArrayList<Double> price, double t, I input);
}
