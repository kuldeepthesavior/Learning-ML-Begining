import unittest

def isEven(x):
  if int(x)%2==0:
    return True
  else:
    return False
    
class TestIsEvenMethod(unittest.TestCase):
  def testEven(self):
    self.assertTrue(isEven(2))
    
    
  def testOdd(self):
    self.assertFalse(isEven(3))
  
  def test_isEven1(self):
    self.assertEqual(isEven(5),True)
    
    
  def test_split(self):
    s = 'hello'
    self.assertEqual(isEven(s), True)
    # check that s.split fails when the separator is not a string
    with self.assertRaises(TypeError):
      self.assertEqual(isEven(s), True)
      
      
t=TestIsEvenMethod()
#t.test_isEven1()
if __name__ == '__main__':
    unittest.main()
