mod vec_own {
    use std::fmt;
    use std::ops; 

    pub struct VecOwn<T>(Vec<T>);

    impl<T> VecOwn<T> {
        pub fn new() -> Self {
            let vec: Vec<T> = Vec::new();
            return VecOwn::<T>(vec);
        }
    }

    impl<T> std::cmp::PartialEq for VecOwn<T> {
        fn eq(&self, other: &VecOwn<T>) -> bool {
            if self.len() != other.len() {
                return false;
            }

            return true;
        }

        fn ne(&self, other: &VecOwn<T>) -> bool {
            if self.len() == other.len() {
                return false;
            }

            return true;
        }
    }

    impl<T: std::clone::Clone> VecOwn<T> {
        pub fn new_from_arr(arr: &[T]) -> Self {
            let vec: Vec<T> = arr.to_vec();
            return VecOwn::<T>(vec);
        }
    }

    impl<T> Default for VecOwn<T> {
        fn default() -> Self {
            return Self::new();
        }
    }

    impl<T: std::clone::Clone> Clone for VecOwn<T> {
        fn clone(&self) -> Self {
            return Self(Vec::<T>::clone(&self.0));
        }
    }

    impl<T> ops::Deref for VecOwn<T> {
        type Target = Vec<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T> ops::DerefMut for VecOwn<T> {
         fn deref_mut(&mut self) -> &mut Self::Target {
             &mut self.0
         }
    }

    impl<T: fmt::Display> fmt::Display for VecOwn<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "[")?;
            for v in self.iter() {
                write!(f, "{}, ", v)?;
            }
            write!(f, "]")?;
            Ok(())
        }
    }

    impl<T: fmt::UpperHex> fmt::UpperHex for VecOwn<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let v_vec: &Vec<T> = &self.0;

            write!(f, "[")?;
            for v in v_vec {
                fmt::UpperHex::fmt(&v, f)?;
                write!(f, ", ")?;
            }
            write!(f, "]")?;
            Ok(())
        }
    }
}