/***********************************https://www.onlinegdb.com/edit/fpgcWleJ5#editor_1*******************************************

                            Online Java Debugger.
                Code, Run and Debug Java program online.
Write your code in this editor and press "Debug" button to debug program.

*******************************************************************************/
import java.nio.file.*;
import java.util.*;

public class Kind2  {
	class Td {
	    public String fileName;
		public boolean t;
		public boolean[] b;
		public int cnt;

		public Td( Path file ) throws Exception {
			// System.out.println(file);
			fileName = file.getFileName().toString();
			if( fileName.startsWith("1") ) {
				t = true;
			}
			List<String> lines = Files.readAllLines(file);
			if( lines.size() != 16 ) throw new Exception("line count must be 16");
			int j = 0;
			b = new boolean[256];
			for( String line : lines ) {
				for( int i=0 ;  i < 16 ; i++ ) {
					if( line.charAt(i) == '■' ) {
						b[j] = true;
						cnt++;
					}
					j++;
				}
			}
			// System.out.println("j=" + j );
			if( j != 256 ) throw new Exception("bit count must be 256");
		}
	}

	public static void main(String args[]) {
		System.out.println("Hello");
        Main main = new Main();
        
        main.doAi();
	}
	
	public double predict( boolean[] b, double[] weights, double bias) {
	    double predition = bias;
	    for( int i = 0; i <b.length ; i++) {
	        predition += (b[i] ? 1.0 : 0.0) * weights[i];  
	    }
	    return predition; 
	}
	
	public void doAi() {

		List<Td> tds = new ArrayList<Td>();
		try {
			try (DirectoryStream<Path> files =
				            Files.newDirectoryStream(Path.of(".\\data2"), "*.txt")) {

				for (Path file : files) {
					Td td = new Td(file);
					tds.add(td);
				}
			}
		} catch ( Exception e ) {
			e.printStackTrace();
		}
		
		double[] weights = new double[256];
		for( double w : weights) w = 0.0;
		
		double bias = 0.0;
		
		for( Td td : tds ) {
		    System.out.println("file=" + td.fileName + " t=" + td.t + " predict=" + predict(td.b, weights, bias)
		        + " " + td.cnt );
		}
		
		double rate = 0.04;
		int epoch = 600;
		
		for( int i = 0 ; i < epoch ; i++ ) {
		    double total_loss = 0.0;
		    Collections.shuffle(tds);
		    
		    for( Td td : tds ) {
		        
		        double predict = predict(td.b, weights, bias);
		        double target = td.t ? 1.0 : 0.0;
		        double loss =  Math.pow(    (predict - target), 2.0);
		        total_loss += loss;
		        
		        double gradient = 2.0 * (predict - target);
		        
		        for( int j = 0 ; j < td.b.length ; j++ ) {
		            double value = td.b[j] ? 1.0 : 0.0;
		            weights[j] -= rate * gradient * value; 
		        }
		        bias -= rate * gradient;
		    }
		    
		    if( i % 100 == 0 )
		        System.out.println((i+1) + " " + (total_loss/tds.size()) );
		}
		
				
		for( Td td : tds ) {
		    System.out.println("file=" + td.fileName + " t=" + td.t + " predict=" + predict(td.b, weights, bias)
		        + " " + td.cnt );
		}
		
	}
}

