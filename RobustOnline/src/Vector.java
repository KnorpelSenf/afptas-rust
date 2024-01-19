import java.util.HashMap;
import java.util.Iterator;

/**
 * This represents a spare vector, where only few indices are non-zero
 * and the stored values are fractional
 *
 * @param <T> the type of indices used
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class Vector<T> implements Iterable<T>{
    
    /**
     * An HashMap containing the indices which are non-zero
     */
    HashMap<T,Double> map;
    
    /**
     * Creates a new Vector containing just zeroes
     *
     */
    public Vector(){
        map = new HashMap<T,Double>();
    }
    
    /**
     * Creates a new Vector as copy of another one
     *
     * @param v the Vector to be copied
     */
    public Vector(Vector<T> v){
        map = new HashMap<T,Double>();
        for (T t: v){
            put(t,v.get(t));
        }
    }
    
    /**
     * Returns the value stored under a index
     *
     * @param t the index to look at
     * @return the number which is stored under t or <code>null</null>
     * if no number is stored
     */
    public Double get(T t){
        return map.get(t);
    }

    /**
     * Creates a new Vector as the sum of this vector and another one
     *
     * @param o the vector to be added to this vector
     * @return a new vector, which is the sum of this and o
     */
    public Vector<T> add(Vector<T> o){
        Vector<T> res = new Vector<T>(o);

        // Add all entries to the new vector
        for (T t: this){
            if (o.map.containsKey(t)){
                res.put(t,get(t)+o.get(t));
            }
            else{
                res.put(t,get(t));
            }
        }
        return res;
    }

    /**
     * Creates a new Vector as a scaled copy of this 
     *
     * @param tau the value to scale the vector
     * @return a new vector, which is tau times this vector
     */
    public Vector<T> scale(double tau){
        Vector<T> res = new Vector<T>(this);
        for(T t: this){
            res.put(t,get(t)*tau);
        }
        return res;
    }
    
    /**
     * Add an index and a value to the Vector
     * 
     * @param t the index to store the value under
     * @param n the value to store
     */
    public void put(T t, double n){
        if (n==0){
            map.remove(t);
        }
        else{
            map.put(t,n);
        }
    }
    
    /**
     * Iterates over the indices which contain non-zero entries
     *
     * @return an {@link Iterator} iterating over the indices
     */
    public Iterator<T> iterator(){
        return map.keySet().iterator();
    }
    
    /**
     * Returns the accumulated values of the stored indices
     *
     * @return the sum of the values
     */
    public double sum(){
        double sum = 0;
        for (T t: this){
            sum = sum + map.get(t);
        }
        return sum;
    }
    
    /**
     * Returns the number of non-zero entries
     *
     * @return the number of non-zero entries
     */
    public int getSize(){
        return map.size();
    }
}



