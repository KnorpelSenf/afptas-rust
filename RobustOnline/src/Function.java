import java.util.ArrayList;

/**
 * This represents a function from a {@link Vector} containing T to a
 * number
 *
 * @param <T> the domain of the function
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public abstract class Function<T>{

    /**
     * Creates a new Function
     *
     */
    public Function(){}
    
    /**
     * Apply this function to an {@link Vector} containing T
     * 
     * @param param the parameter of the function
     * @return the value of the function
     */
    public abstract double apply(Vector<T> param);
}
