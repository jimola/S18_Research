import Control.Monad.Except
-- This is the type to represent length calculation error.
data LengthError = EmptyString  -- Entered string was empty.
          | StringTooLong Int   -- A string is longer than 5 characters.
                                -- Records a length of the string.
          | OtherError String   -- Other error, stores the problem description.

-- Converts LengthError to a readable message.
instance Show LengthError where
  show EmptyString = "The string was empty!"
  show (StringTooLong len) =
      "The length of the string (" ++ (show len) ++ ") is bigger than 5!"
  show (OtherError msg) = msg

-- For our monad type constructor, we use Either LengthError
-- which represents failure using Left LengthError
-- or a successful result of type a using Right a.
type LengthMonad = Either LengthError

main = do
  putStrLn "Please enter a string:"
  s <- getLine
  reportResult (calculateLength s)

-- Wraps length calculation to catch the errors.
-- Returns either length of the string or an error.
-- catchError : m a -> (e -> m a)
calculateLength :: String -> LengthMonad Int
calculateLength s = (calculateLengthOrFail s) `catchError` Left

-- Attempts to calculate length and throws an error if the provided string is
-- empty or longer than 5 characters.
-- The processing is done in Either monad.
-- throwError : e -> m a
calculateLengthOrFail :: String -> LengthMonad Int
calculateLengthOrFail [] = throwError EmptyString
calculateLengthOrFail s | len > 5 = throwError (StringTooLong len)
                        | otherwise = return len
  where len = length s

-- Prints result of the string length calculation.
reportResult :: LengthMonad Int -> IO ()
reportResult (Right len) = putStrLn ("The length of the string is " ++ (show len))
reportResult (Left e) = putStrLn ("Length calculation failed with error: " ++ (show e))



