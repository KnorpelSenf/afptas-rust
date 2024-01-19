import java.io.*;
import javax.xml.parsers.*;
import org.xml.sax.*;
import org.xml.sax.helpers.DefaultHandler;
import java.util.ArrayList;

/**
 * Reads a xml file describing a series of online bin packing problems
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class ParseInput extends DefaultHandler{
    
    /**
     * Stores the problem in a matrix
     */
    public double[][] sizes;
    
    /**
     * The current instance
     */
    int i;

    /**
     * The current number of the item in instance i
     */
    int j;
    
    /**
     * Creates a new ParseInput instance
     *
     * @param file the xml file describing the problem
     * @param imax the number of instances
     * @param jmax the maximal size of the instances
     */
    public ParseInput(String file, int imax, int jmax){
        try{
            i = 0;
            j = 0;
            sizes = new double[imax][jmax];
            SAXParser saxParser = SAXParserFactory.newInstance().newSAXParser();
            saxParser.parse( new File(file), this);
        }
        catch(Exception e){
            System.out.println(e);
            

        }
    }

    /**
     * Parse the file and create the instance
     *
     * @param namespaceURI The namespace of the element
     * @param localName The local name of the element
     * @param qName The fully qualified name of the element
     * @param attrs The attributes of the element
     * @exception SAXException if an error occurs
     */
    public void startElement( String namespaceURI,
                              String localName,   
                              String qName,       
                              Attributes attrs )
        throws SAXException
    {
        // use the qualified name if local name is empty
        String eName = ( "".equals( localName ) ) ? qName : localName;
        // we found a number
        if (eName.equals("number")) {
            double size = Double.parseDouble(attrs.getValue(0));
            sizes[i][j] = size;
            j++;
        }
        // we found a new instance
        if (eName.equals("list")) {
            i++;
            j = 0;
        }

    }

}
