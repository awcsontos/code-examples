%%

%class scanner
%unicode
%line
%column
%byaccj

%{

	/* store a reference to the parser object */
	private Parser yyparser;

	/* constructor taking an additional parser */
	public scanner (java.io.Reader r, Parser yyparser) {
		this (r);	
		this.yyparser = yyparser;
	}

	/* return the current line number. We need this
	   because yyline is made private and we don't have
	   a mechanism like extern in C. */
	public int getLine() {
		return yyline;
	}

%}

%%
"="		{return Parser.EQUAL;}
"("		{return Parser.LPAREN;}
")"		{return Parser.RPAREN;}
"["		{return Parser.LARR;}
"]"		{return Parser.RARR;}
"def"		{return Parser.FUNCDEF;}
"fed"		{return Parser.FUNCEND;}
"if"		{return Parser.IF;}
"while"		{return Parser.WHILE;}
"elihw"		{return Parser.ENDWHILE;}
"or"		{return Parser.OR;}
"and"		{return Parser.AND;}
"print"		{return Parser.PRINT;}
"true"		{return Parser.TRUE;}
"false"		{return Parser.FALSE;}
"<"		{return Parser.LSTN;}
">"		{return Parser.GTTN;}
":"		{return Parser.FUNCSRT;}
"\r|\n|\rn"	{return Parser.CR;}
","		{return Parser.COMMA;}
[0-9]+		{yyparser.yylval = new ParserVal(Integer.parseInt(yytext()));return Parser.INT;}
[a-z]		{yyparser.yylval = new ParserVal(yytext()); return Parser.ID;}
[ \t]		{;}