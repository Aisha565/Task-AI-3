from google import genai
from google.genai.errors import ClientError

client = genai.Client(api_key="AQ.Ab8RN6KC7fKvpnMg7IMDzmup2QpvqwdRiN1Bs8_RxjtHSmsBVw")


def generate_answer(prompt):

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except ClientError as e:

        # Quota exceeded
        if "429" in str(e):

            return (
                "LLM quota exceeded temporarily. "
                "Please try again later."
            )

        return "LLM generation failed."

    except Exception:

        return "Unexpected AI generation error."