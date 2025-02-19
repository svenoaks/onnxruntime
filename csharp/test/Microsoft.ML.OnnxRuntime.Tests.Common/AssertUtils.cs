﻿using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
  internal static class AssertUtils
  {

    public static void IfThrowsCheckException<T>(Action action, string failureFeedbackMessage, string expectedExceptionMessage = null) where T : Exception
    {
      try
      {
        action();

      }
      catch (T ex)
      {
        if (expectedExceptionMessage == null)
        {
          return;
        }
        else
        {
          Assert.Contains(expectedExceptionMessage, ex.Message);
        }
      }
      catch (Exception ex)
      {
        Assert.Fail($"{failureFeedbackMessage}\nExpected {typeof(T).Name} but got {ex.GetType().Name}. {failureFeedbackMessage}");
      }
    }
  }
}
