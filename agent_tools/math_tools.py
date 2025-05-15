# Ensure mpmath is available
import mpmath

# Import Langchain modules
from langchain_core.tools import Tool


# Define the tool for Stirling's approximation
def stirling_approximation_factorial(n_str: str) -> str:
    """
    Calculates Stirling's approximation for n!.
    Handles potentially large n as a string input.
    Returns a string representation of the approximation, always in scientific
    notation for results >= 1e10.
    """
    try:
        n = int(n_str)
        if n < 0:
            return "Error: Factorial is not defined for negative numbers."
        if n == 0:
            return "1"  # 0! = 1
        # Use mpmath for precision with potentially large numbers
        with mpmath.workdps(
                50):  # Work with 50 decimal places of precision
            log_n_factorial = (mpmath.mpf(n) + 0.5) * mpmath.log(
                mpmath.mpf(n)) - mpmath.mpf(n) + 0.5 * mpmath.log(
                    2 * mpmath.pi)
            # The result is log_e(n!), so n! = exp(log_n_factorial)
            approx_val = mpmath.exp(log_n_factorial)

            # Ensure scientific notation for large numbers
            if approx_val >= 1e10:
                # Format to scientific notation with a reasonable number of significant figures
                return mpmath.nstr(approx_val, n=10, min_fixed=-1, max_fixed=-1, show_zero_exponent=False)
            else:
                # For smaller numbers, standard representation is fine, but ensure it's a string
                return mpmath.nstr(approx_val, n=15) # More precision for smaller numbers if needed
    except ValueError:
        return "Error: Input must be a valid integer."
    except Exception as e:
        return f"Error during Stirling's approximation: {str(e)}"


# Create a Langchain Tool for this
stirling_tool = Tool(
    name="stirling_approximation_for_factorial",
    func=stirling_approximation_factorial,
    description="Calculates Stirling's approximation for n! (factorial of n). Use this for large n (e.g., n > 70) or if direct calculation fails due to resource limits. Input should be a string representing the integer n. The result will be in scientific notation for very large numbers."
)
