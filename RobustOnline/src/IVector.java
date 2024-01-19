import java.util.HashMap;
import java.util.Iterator;

/**
 * This represents a spare vector, where only few indices are non-zero
 * and the stored values are integral
 *
 * @param <T> the type of indices used
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class IVector<T> implements Iterable<T>{
    
    /**
     * An HashMap containing the indices which are non-zero
     */
    HashMap<T,Integer> map;
    
    /**
     * Creates a new IVector containing just zeroes
     *
     */
    public IVector(){
        map = new HashMap<T,Integer>();
    }
    
    /**
     * Creates a new IVector as copy of another one
     *
     * @param v the IVector to be copied
     */
    public IVector(IVector<T> v){
        map = new HashMap<T,Integer>();
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
    public Integer get(T t){
        return map.get(t);
    }

    /**
     * Creates a new IVector as the sum of this vector and another one
     *
     * @param o the vector to be added to this vector
     */
    public IVector<T> add(IVector<T> o){
        IVector<T> res = new IVector<T>(o);

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
     * Add an index and a value to the IVector
     * 
     * @param t the index to store the value under
     * @param n the value to store
     */
    public void put(T t, int n){
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

    /**
     * Converts the integral vector into a fractional vector
     * @return the fractional vector containing the same indices
     */
    public Vector<T> toVector(){

        Vector<T> v = new Vector<>();

        for (T t: this){
            v.put(t,(double)map.get(t));
        }
        return v;
    }
    
}




