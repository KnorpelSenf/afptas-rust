import java.util.HashMap;
import java.util.Iterator;

/**
 * This represents a collection which associates values with integers
 *
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 * @param <T> the type of values stored
 */
public class Pattern<T> implements Iterable<T>{

    /**
     * The HashMap storing the values
     */
    HashMap<T,Integer> items;
    
    /**
     * Creates a new Pattern which does not store anything
     *
     */
    public Pattern(){
        items = new HashMap<T,Integer>();
    }

    /**
     * Creates a new Pattern as a copy of a pattern
     *
     * @param p the pattern to be copied
     */
    public Pattern(Pattern<T> p){
        items = new HashMap<T,Integer>();
        for (T t: p){
            items.put(t,p.get(t));
        }
    }
    
    /**
     * Adds a value and a number to the stored values
     * 
     * @param a the value to be stored as key
     * @param n the integers to be stored under the key
     */
    public void put(T a,int n){
        if(n==0){
            items.remove(a);
        }
        else{
            items.put(a,n);
        }
    }

    /**
     * Either adds a value and a number to the stored values or
     * increases the number if the value is already present
     *
     * @param a the value to be stored as key
     * @param n the integers to be stored under the key
     */
    public void add(T a, int n){
        if (items.containsKey(a)) {
            put(a,items.get(a)+n);
        }
        else{
            put(a,n);
        }
    }
        
    /**
     * Returns the value stored under a index
     *
     * @param a the index to look at
     * @return the integer which is stored under t or <code>null</null>
     * if no number is stored
     */
    public Integer get(T a){
        return items.get(a);
    }
    
    /**
     * Either adds a value and a <code>1</code> or adds <code>1</code>
     * if it is already stored
     *
     * @param a the index to be added
     */
    public void addOrIncrease(T a){
        add(a,1);
    }
    
    /**
     * Compares this with another object
     * 
     * @param o the object to compare with
     * @return <code>true</code> if the other object is a pattern which
     *         stores the same elements and the same integers;
     *         <code>false</code> otherwise.
     */
    public boolean equals(Object o){
        boolean b = false;
        if (o instanceof Pattern){

            // We can not check generic conversion
            @SuppressWarnings("unchecked")
            Pattern<T> p = (Pattern<T>) o;
            if (p.items.keySet().equals(items.keySet())){
                b = true;
                for (T t: items.keySet()){
                    if (p.get(t)!=get(t)){
                        b = false;
                    }
                }
            }
        }
        return b;
    }

    /**
     * Computes the hash code of the pattern
     *
     * @return a linear combinator of the hash codes of the stored values
     */
    public int hashCode(){
        int sum = 0;
        for (T t: items.keySet()){
            sum = items.get(t)*t.hashCode();
        }
        return sum;
    }

    /**
     * Iterates over the indices which contain non-zero entries
     *
     * @return an {@link Iterator} iterating over the indices
     */
    public Iterator<T> iterator(){
        return items.keySet().iterator();
    }

    /**
     * Tests whether the pattern is empty
     *
     * @return true if no value is stored in the pattern
     */
    public boolean isEmpty(){
        return items.keySet().isEmpty();
    }
}





