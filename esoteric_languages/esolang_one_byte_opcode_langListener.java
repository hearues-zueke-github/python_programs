// Generated from esolang_one_byte_opcode_lang.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link esolang_one_byte_opcode_langParser}.
 */
public interface esolang_one_byte_opcode_langListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link esolang_one_byte_opcode_langParser#function}.
	 * @param ctx the parse tree
	 */
	void enterFunction(esolang_one_byte_opcode_langParser.FunctionContext ctx);
	/**
	 * Exit a parse tree produced by {@link esolang_one_byte_opcode_langParser#function}.
	 * @param ctx the parse tree
	 */
	void exitFunction(esolang_one_byte_opcode_langParser.FunctionContext ctx);
	/**
	 * Enter a parse tree produced by {@link esolang_one_byte_opcode_langParser#variable}.
	 * @param ctx the parse tree
	 */
	void enterVariable(esolang_one_byte_opcode_langParser.VariableContext ctx);
	/**
	 * Exit a parse tree produced by {@link esolang_one_byte_opcode_langParser#variable}.
	 * @param ctx the parse tree
	 */
	void exitVariable(esolang_one_byte_opcode_langParser.VariableContext ctx);
	/**
	 * Enter a parse tree produced by {@link esolang_one_byte_opcode_langParser#assign}.
	 * @param ctx the parse tree
	 */
	void enterAssign(esolang_one_byte_opcode_langParser.AssignContext ctx);
	/**
	 * Exit a parse tree produced by {@link esolang_one_byte_opcode_langParser#assign}.
	 * @param ctx the parse tree
	 */
	void exitAssign(esolang_one_byte_opcode_langParser.AssignContext ctx);
	/**
	 * Enter a parse tree produced by {@link esolang_one_byte_opcode_langParser#number_decimal}.
	 * @param ctx the parse tree
	 */
	void enterNumber_decimal(esolang_one_byte_opcode_langParser.Number_decimalContext ctx);
	/**
	 * Exit a parse tree produced by {@link esolang_one_byte_opcode_langParser#number_decimal}.
	 * @param ctx the parse tree
	 */
	void exitNumber_decimal(esolang_one_byte_opcode_langParser.Number_decimalContext ctx);
	/**
	 * Enter a parse tree produced by {@link esolang_one_byte_opcode_langParser#operation}.
	 * @param ctx the parse tree
	 */
	void enterOperation(esolang_one_byte_opcode_langParser.OperationContext ctx);
	/**
	 * Exit a parse tree produced by {@link esolang_one_byte_opcode_langParser#operation}.
	 * @param ctx the parse tree
	 */
	void exitOperation(esolang_one_byte_opcode_langParser.OperationContext ctx);
}