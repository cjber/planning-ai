# build_quarto_doc(doc_title, out)
#
# d = [
#     i for i in out["generate_final_summary"]["summaries_fixed"] if i["iteration"] == 4
# ][0]
# d["document"]
#
# h = [
#     i["summary"].summary
#     for i in out["generate_final_summary"]["hallucinations"]
#     if i["document"] == d["document"]
# ]
#
# e = [
#     i["hallucination"].explanation
#     for i in out["generate_final_summary"]["hallucinations"]
#     if i["document"] == d["document"]
# ]
#
# test = {
#     "document": d["document"],
#     "final_summary": d["summary"].summary,
#     "attempts": h,
#     "reasoning": e,
# }
#
# print(f"Document:\n\n{test['document']}\n\n")
# print(f"Final:\n\n{test['final_summary']}\n\n")
# print("Attempts: \n\n*", "\n\n* ".join(test["attempts"]), "\n\n")
# print("Reasoning: \n\n*", "\n\n* ".join(test["reasoning"]), "\n\n")
