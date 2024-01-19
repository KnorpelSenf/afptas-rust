/**
 * This represents a tree-like structure containing an {@link KPItem}
 * and a reference to another StoreTree
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class StoreTree{

    /**
     * The stored item
     */
    KPItem item;
    
    /**
     * The children of the tree
     */
    StoreTree next;
    
    /**
     * Creates a new StoreTree containing only one {@link KPItem}
     *
     * @param i the KPItem to be stored
     */
    public StoreTree(KPItem i){
        this.item = i;
        next = null;
    }
    
    /**
     * Creates a new StoreTree containing an {@link KPItem} and a children
     *
     * @param i the KPItem to be stored
     * @param o the children to be stored
     */
    public StoreTree(KPItem i, StoreTree o){
        this.item = i;
        next = o;
    }
}
