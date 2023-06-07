%{
import java.io.*;
import java.util.*;
%}
%left CR
%token IF WHILE RARR LARR EQUAL OR AND CR LPAREN RPAREN LSTN GTTN FUNCDEF FUNCSRT FUNCEND PRINT ENDWHILE TRUE FALSE COMMA
%token <sval> ID TRUE FALSE
%token <ival> INT
%type <ival> boolCmp boolOp expr 
%type <sval> param param_list statementList statement

%%
lines	:	lines line
	|	line
	;

line	:	FUNCDEF ID LPAREN RPAREN FUNCSRT statementList {if(currScope.equals("global")){currScope = $2;} else{yyerror("Cannot declare function inside function!");};} 
	|	FUNCDEF ID LPAREN param_list RPAREN FUNCSRT statementList {if(currScope.equals("global")){currScope = $2;}else{yyerror("Cannot declare function inside function!");};} 
	|	FUNCEND {currScope = "global";}
	|	IF boolCmp FUNCSRT CR statementList CR {if($2 == 1){$5.toString();};}
	|	IF boolOp FUNCSRT CR statementList CR {if($2 == 1){$5.toString();};}	
	|	WHILE boolCmp FUNCSRT statementList ENDWHILE {while($2 == 1){$4.toString();};}
	|	WHILE boolOp FUNCSRT statementList ENDWHILE {while($2 == 1){$4.toString();};}
	|	statementList	
	;

statementList:	statementList CR statement
	|	statement		
	;

statement:	PRINT param CR	{System.out.println($2.toString());}
	|	PRINT ID LARR INT RARR CR	{System.out.println(lookupArrayVal($2, $4));}
	|	ID EQUAL INT CR {assign($1, currScope); assignInt($1, $3);}
	|	ID EQUAL ID CR {assign($1, currScope);  assignInt($1, lookupInt($3));}
	|	ID LARR INT RARR EQUAL expr CR{assign(("["+$1.toString() + "]"), currScope); assignArrayVal($1, $3, $6);}
	|	ID EQUAL ID LARR INT RARR CR {assign($1, currScope); assignInt($1, lookupArrayVal($3, $5));}
	;

boolCmp : 	ID EQUAL EQUAL ID {if(lookupInt($1) == lookupInt($4)) {$$ = 1;} else{$$ = 0;};}
	|	ID LSTN EQUAL ID {if(lookupInt($1) <= lookupInt($4)) {$$ = 1;} else{$$ = 0;};}
	|   	ID GTTN EQUAL ID {if(lookupInt($1) >= lookupInt($4)) {$$ = 1;} else{$$ = 0;};}
	|	ID LSTN ID {if(lookupInt($1) < lookupInt($3)) {$$ = 1;} else{$$ = 0;};}
	|	ID GTTN ID {if(lookupInt($1) > lookupInt($3)) {$$ = 1;} else{$$ = 0;};}
	|	ID EQUAL EQUAL INT {if(lookupInt($1) == $4) {$$ = 1;} else{$$ = 0;};}
	|	ID LSTN EQUAL INT {if(lookupInt($1) <= $4) {$$ = 1;} else{$$ = 0;};}
	|   	ID GTTN EQUAL INT {if(lookupInt($1) >= $4) {$$ = 1;} else{$$ = 0;};}
	|	ID LSTN INT {if(lookupInt($1) < $3) {$$ = 1;} else{$$ = 0;};}
	|	ID GTTN INT {if(lookupInt($1) > $3) {$$ = 1;} else{$$ = 0;};}

	;

boolOp  : 	TRUE OR TRUE {$$ = 1;}
	|	TRUE AND TRUE {$$ = 1;}
	|	TRUE OR FALSE {$$ = 1;}
	|	TRUE AND FALSE {$$ = 0;}
	|	FALSE OR TRUE {$$ = 1;}
	|	FALSE AND TRUE {$$ = 0;}
	|	FALSE OR FALSE {$$ = 0;}
	|	FALSE AND FALSE {$$ = 0;}
	;

expr	:	'-' FALSE {$$ = 1;}
	|	'-' TRUE {$$ = 0;}
	|	'-' INT {$$ = -1 * ($2);}
	|	INT '+' INT{$$ = $1+$3;}
 	|	INT '-' INT {$$ = $1-$3;}
 	|	INT '*' INT {$$ = $1*$3;}
 	|	INT '/' INT { if ($3!=0){ $$ = $1/$3; }else { System.out.print("Error: divide by Zero"); } }
 	|	param {$$=1;}
 	;

param_list:	param 
	|	param_list COMMA param
	;


param	:	INT	{$$ = Integer.toString($1);}	
	|	ID	{$$ = $1;}
%%

/* Byacc/J expects a member method int yylex(). We need to provide one
   through this mechanism. See the jflex manual for more information. */

	/* reference to the lexer object */
	private scanner lexer;

	String currScope = "global";
	private HashMap symbol_table = new HashMap();
	private HashMap stored_values = new HashMap();	private HashMap<String, ArrayList<Integer>> arrayValues = new HashMap<String, ArrayList<Integer>>();

	/* interface to the lexer */
	private int yylex() {
		int retVal = -1;
		try {
			retVal = lexer.yylex();
		} catch (IOException e) {
			System.err.println("IO Error:" + e);
		}
		return retVal;
	}
	
	/* error reporting */
	public void yyerror (String error) {
		System.err.println("Error : " + error + " at line " + lexer.getLine());
		System.err.println("Something broke!");
	}

	/* this method stores an id on the symbol table with its associated scope */
	public void assign (String id, String scope) {
		symbol_table.put (id, scope);
	}

	public String lookup (String id) {
		return (symbol_table.get(id).toString());
	}

	/* this method stores an id, int pair for easy variable storage. */
	public void assignInt (String id, int val) {
		stored_values.put(id, val);
	}

	public int lookupInt (String id) {
		return ((Integer)(stored_values.get(id)));
	}

	/* this method stores a value at the given array and index in that array */
	public void assignArrayVal (String name, int val, int index) 
	{
		if(arrayValues.get(name) == null)
		{
			ArrayList<Integer> arrayVals = new ArrayList<Integer>();
			arrayVals.add(index, val);
			arrayValues.put(name, arrayVals);
		}
		else
		{
			ArrayList<Integer> arrayVals = arrayValues.get(name);
			arrayVals.add(index, val);
			arrayValues.put (name, arrayVals);
		}
	}

	public int lookupArrayVal (String name, int index) {
		return ((Integer)(arrayValues.get(name).get(index)));

	}
	
	
	/* constructor taking in File Input */
	public Parser (Reader r) {
		lexer = new scanner (r, this);
	}

	public static void main (String [] args) throws IOException {
		Parser yyparser = new Parser(new FileReader(args[0]));
		yyparser.yyparse();
	}
