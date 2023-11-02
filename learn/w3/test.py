import unittest

class Grade:
    def evaluate(self, score: int):
        return "F"

class TestGrade(unittest.TestCase):
    def test_level(self, score: int):
        grade = Grade()
        self.assertEqual(grade.evaluate(-100), "F")
        self.assertEqual(grade.evaluate(31), "C")
        self.assertEqual(grade.evaluate(15), "D")

        self.assertEqual(grade.evaluate(-1), "F")
        self.assertEqual(grade.evaluate(0), "D")
        self.assertEqual(grade.evaluate(1), "D")

        self.assertEqual(grade.evaluate(25), "D")
        self.assertEqual(grade.evaluate(58), "B")
        self.assertEqual(grade.evaluate(35), "C")

        self.assertEqual(grade.evaluate(29), "D")        
        self.assertEqual(grade.evaluate(30), "C")
        self.assertEqual(grade.evaluate(31), "C")

        self.assertEqual(grade.evaluate(32), "C")
        self.assertEqual(grade.evaluate(69), "A")
        self.assertEqual(grade.evaluate(55), "B")
        
        self.assertEqual(grade.evaluate(49), "C")
        self.assertEqual(grade.evaluate(50), "B")
        self.assertEqual(grade.evaluate(51), "B")

        self.assertEqual(grade.evaluate(61), "B")
        self.assertEqual(grade.evaluate(85), "F")
        self.assertEqual(grade.evaluate(68), "A")
        
        self.assertEqual(grade.evaluate(64), "B")
        self.assertEqual(grade.evaluate(65), "A")
        self.assertEqual(grade.evaluate(66), "A")

        self.assertEqual(grade.evaluate(69), "B")
        self.assertEqual(grade.evaluate(70), "A")
        self.assertEqual(grade.evaluate(71), "F")

if __name__ == '__main__':
    unittest.main()