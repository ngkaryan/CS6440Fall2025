

4\. The notebook uses \*\*RAG retrieval\*\* and the \*\*Gemini LLM\*\* to generate answers.



---



\## Step 6: Adding New Data or Updating the Notebook



1\. \*\*Add new FHIR JSON files\*\* to `synthea\_output/` locally.

2\. Make any notebook updates as needed.

3\. Compress the \*\*entire folder\*\* again into a new ZIP (e.g., `fhir-gemini-rag-voila-v2.zip`).

4\. Re-upload the new ZIP to your HuggingFace Space \*\*Files tab\*\*.

5\. HuggingFace will overwrite old files and refresh the app automatically.



> ⚠️ Important: Always include \*\*all files\*\* in the ZIP, not just new ones, otherwise old files may be lost.



---



\## Step 7: Optional Tips



\- \*\*Secrets\*\*: Always use HuggingFace Secrets for API keys.

\- \*\*Free tier limits\*\*: The Space may sleep after inactivity; refresh the browser to wake it up.

\- \*\*Data size\*\*: Keep ZIP files small (<100 MB recommended) for free tier.



---



\## Notes



\- Supports \*\*all FHIR resources\*\* via PyDantic models.

\- Users can query the dataset in \*\*natural language\*\*.

\- Designed to work with \*\*Synthea-generated synthetic FHIR datasets\*\*.

\- Git and SSH access are \*\*not required\*\*.



---



Enjoy exploring your FHIR dataset with LLM-powered natural language queries!



