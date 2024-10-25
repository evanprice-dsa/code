// java connection to azure sql

// imports
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.Statement;
import java.sql.Types;
import java.util.Scanner;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.DriverManager;

public class Price_Evan_HW3_Problem2 {
	
    // database credentials
    final static String HOSTNAME = "";
    final static String DBNAME = "";
    final static String USERNAME = "";
    final static String PASSWORD = "";

    // database connection string
    final static String URL = String.format("jdbc:sqlserver://%s:1433;database=%s;user=%s;password=%s;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;",
            HOSTNAME, DBNAME, USERNAME, PASSWORD);
    
	// program starts here
    public static void main(String[] args) throws SQLException {
		
    	// read in user input on repeat loop
    	Scanner scanner = new Scanner(System.in);
        
    	// connect to azure 
    	try (Connection connection = DriverManager.getConnection(URL)) {
            
    		// breakout variable
    		boolean exit = false;

            // only break out when user specifies
            while (!exit) {
                System.out.println("1. Option 1 new faculty");	// by department id
                System.out.println("2. Option 2 new faculty");	// by particular department id
                System.out.println("3. Display all faculty");	// view all
                System.out.println("4. Quit");					// exit
                System.out.print("Choose an option: ");			// request input
                int option = scanner.nextInt();					// get input
                scanner.nextLine(); 							// and newline
                
                switch (option) {			// user input options are here
                    case 1:
                        insertFaculty1(connection, scanner);
                        break;
                    case 2:
                        insertFaculty2(connection, scanner);
                        break;
                    case 3:
                        displayFaculty(connection);
                        break;
                    case 4:
                        exit = true;
                        System.out.println("Exit program");
                        break;
                    default:
                        System.out.println("Invalid input");
                        break;
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    // option 1 function
    private static void insertFaculty1(Connection connection, Scanner scanner) throws SQLException {
        // only three user inputs for option one
    	System.out.print("Enter faculty id: ");		// number as integer
        int fid = scanner.nextInt();
        scanner.nextLine();
        
        System.out.print("Enter faculty name: ");	// name as string
        String fname = scanner.nextLine();
        
        // no newline on user input
        System.out.print("Enter department id: ");	// department as integer
        int deptid = scanner.nextInt();

        // set up the stored procedure 
        CallableStatement stmt = connection.prepareCall("{call InsertFaculty1(?, ?, ?, ?)}");
        stmt.setInt(1, fid);								// inputs
        stmt.setString(2, fname);							//
        stmt.setInt(3, deptid);								//
        stmt.registerOutParameter(4, Types.REAL); 			// but salary is output

        // run stored procedure
        stmt.execute();

        // retrieve custom salary and output
        float salary = stmt.getFloat(4);
        System.out.println("Salary: " + salary);
        System.out.println("");
    }

    // option 2 function
    private static void insertFaculty2(Connection connection, Scanner scanner) throws SQLException {
        // four inputs for option two
    	System.out.print("Enter faculty id: ");		// number as integer
        int fid = scanner.nextInt();
        scanner.nextLine();
        
        System.out.print("Enter faculty name: ");	// name is string
        String fname = scanner.nextLine();
        System.out.print("Enter department id: ");	// department id as integer
        int deptid = scanner.nextInt();
        											// excluded department id as integer
        System.out.print("Enter department id to exclude: ");
        int excludeDeptid = scanner.nextInt();

        // set up the stored procedure
        CallableStatement stmt = connection.prepareCall("{call InsertFaculty2(?, ?, ?, ?, ?)}");
        stmt.setInt(1, fid);						// inputs
        stmt.setString(2, fname);					//
        stmt.setInt(3, deptid);						//
        stmt.setInt(4, excludeDeptid);				//
        stmt.registerOutParameter(5, Types.REAL); 	// salary is output

        // run stored procedure
        stmt.execute();

        // receive custom salary and output
        float salary = stmt.getFloat(5);
        System.out.println("Salary: " + salary);
        System.out.println("");
    }

    // display all faculty function
    private static void displayFaculty(Connection connection) throws SQLException {
        // sql query
    	String query = "SELECT * FROM Faculty";
        
    	try (Statement stmt = connection.createStatement();
        	// put output in a result set object
    		ResultSet rs = stmt.executeQuery(query)) {
            
    		// display columns
    		System.out.println("fid | fname | deptid | salary");
            
            // output all rows
            while (rs.next()) {
                System.out.println(rs.getInt("fid") + " | " + rs.getString("fname") + " | " + rs.getInt("deptid") + " | " + rs.getFloat("salary"));
            }
            
            // newline at the end for output readability
            System.out.println("");
        }
    }
}


/*
 STORED PROCEDURES SQL FILE  
 -- first stored procedure
-- option 1
-- set salary by assigned department workflow calculation
CREATE OR ALTER PROCEDURE InsertFaculty1
    @fid INT,				-- inputs
    @fname VARCHAR(64),		--
    @deptid INT,			--
    @salary REAL OUTPUT		-- output
AS
BEGIN
    -- assigned department average salary calculation
    DECLARE @avgSalary REAL;
    SELECT @avgSalary = AVG(salary) 
    FROM Faculty
    WHERE deptid = @deptid;

    -- logic to calculate custom salary and save value
    -- three options
    -- IF 
    -- 90% of department average salary if that amount is greater than fifty thousand
    -- ELSE IF
    -- department average salary if that amount is less than thirty thousand
    -- ELSE
    -- set the salary at department average salary
    
    IF @avgSalary > 50000				
        SET @salary = 0.9 * @avgSalary;
    ELSE IF @avgSalary < 30000
        SET @salary = @avgSalary;
    ELSE
        SET @salary = 0.8 * @avgSalary;

    -- sql insert statement
    INSERT INTO Faculty (fid, fname, deptid, salary)
    VALUES (@fid, @fname, @deptid, @salary);
END;

-- second stored procedure
-- option 2
-- set salary by average of salaries save for excluded department id
-- easier implementation
CREATE OR ALTER PROCEDURE InsertFaculty2
    @fid INT,				-- inputs
    @fname VARCHAR(64),		--
    @deptid INT,			--
    @excludeDeptid INT,		--
    @salary REAL OUTPUT		-- output
AS
BEGIN
    -- excluded department average salary calculation
    -- average of all salaries save for the excluded department
    DECLARE @avgSalaryExcludeDept REAL;
    SELECT @avgSalaryExcludeDept = AVG(salary) 
    FROM Faculty
    WHERE deptid <> @excludeDeptid;	
	
    -- save value
    SET @salary = @avgSalaryExcludeDept;
    
    -- sql insert statement
    INSERT INTO Faculty (fid, fname, deptid, salary)
    VALUES (@fid, @fname, @deptid, @avgSalaryExcludeDept);
END;
*/
