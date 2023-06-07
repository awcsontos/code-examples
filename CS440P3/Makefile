parser: Parser.java scanner.java
	javac -classpath "" *.java

Parser.java: typeSystem.y
	./byacc.linux -J typeSystem.y

scanner.java: system.jflex 
	java -jar JFlex.jar system.jflex
clean:
	rm *.class *.java
