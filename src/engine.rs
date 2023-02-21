use layout::{
    adt::dag::NodeHandle,
    backends::svg::SVGWriter,
    core::{base::Orientation, geometry::Point, style::StyleAttr},
    std_shapes::shapes::*,
    topo::layout::VisualGraph,
};
use std::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    fmt::Display,
    ops::{Add, AddAssign, Mul, Sub},
    sync::{Arc, Mutex},
};

pub use num_traits::{Float, NumAssignOps, Zero};

#[derive(Debug)]
pub enum Op {
    ADD,
    SUB,
    MUL,
    // DIV,
    POWI(i32),
    TANH,
}

#[derive(Debug)]
struct Value<T: Float + NumAssignOps> {
    data: T,
    children: (Option<Scalar<T>>, Option<Scalar<T>>),
    op: Option<Op>,
    label: String,
    grad: T,
}

impl<T: Float + NumAssignOps> Value<T> {
    fn new(data: T, label: &str) -> Self {
        Self {
            data,
            children: (None, None),
            op: None,
            label: label.to_string(),
            grad: Zero::zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scalar<T: Float + NumAssignOps>(Arc<Mutex<Value<T>>>);

impl Scalar<f32> {
    pub fn new(data: f32, label: &str) -> Self {
        Scalar(Arc::new(Mutex::new(Value::new(data, label))))
    }

    pub fn label(&self, l: &str) {
        let mut value = self.0.lock().unwrap();

        value.label = l.to_string();
    }

    fn cal_grad(&self) {
        let value = self.0.lock().unwrap();

        match value.op {
            Some(Op::ADD) => {
                if let (Some(c1), Some(c2)) = &value.children {
                    let mut v1 = c1.0.lock().unwrap();
                    v1.grad += value.grad;
                    drop(v1);

                    let mut v2 = c2.0.lock().unwrap();
                    v2.grad += value.grad;
                }
            }
            Some(Op::SUB) => {
                if let (Some(c1), Some(c2)) = &value.children {
                    let mut v1 = c1.0.lock().unwrap();
                    v1.grad += value.grad;
                    drop(v1);

                    let mut v2 = c2.0.lock().unwrap();
                    v2.grad += -value.grad;
                }
            }
            Some(Op::MUL) => {
                if let (Some(c1), Some(c2)) = &value.children {
                    let v1 = c1.0.lock().unwrap();
                    let v1_data = v1.data;
                    drop(v1);

                    let mut v2 = c2.0.lock().unwrap();
                    let v2_data = v2.data;
                    v2.grad += v1_data * value.grad;
                    drop(v2);

                    let mut v1 = c1.0.lock().unwrap();
                    v1.grad += v2_data * value.grad;
                }
            }
            Some(Op::POWI(n)) => {
                if let (Some(c), None) = &value.children {
                    let mut v = c.0.lock().unwrap();
                    v.grad += (n as f32 * v.data.powi(n - 1)) * value.grad;
                }
            }
            Some(Op::TANH) => {
                if let (Some(c), None) = &value.children {
                    let mut v = c.0.lock().unwrap();
                    v.grad += (1.0 - value.data.powi(2)) * value.grad;
                }
            }
            None => (),
        }
    }

    pub fn backward(&self) {
        let scalars = self.traverse();

        for s in &scalars {
            let mut v = s.0.lock().unwrap();
            v.grad = 0.0;
        }

        let mut value = self.0.lock().unwrap();
        value.grad = 1.0;
        drop(value);

        for s in scalars {
            s.cal_grad();
        }
    }
}

impl<T: Float + NumAssignOps> Scalar<T> {
    pub fn data(&self) -> T {
        let v = self.0.lock().unwrap();

        v.data
    }

    pub fn set_data(&self, data: T) {
        let mut v = self.0.lock().unwrap();

        v.data = data;
    }

    pub fn grad(&self) -> T {
        let v = self.0.lock().unwrap();

        v.grad
    }

    pub fn traverse(&self) -> Vec<Self> {
        let mut nodes = vec![self.clone()];
        let mut pointer = 0;

        while nodes.len() > pointer {
            let node = nodes[pointer].0.clone();
            let node = node.lock().unwrap();

            match &node.children {
                (Some(c1), Some(c2)) => {
                    nodes.push(c1.clone());
                    nodes.push(c2.clone());
                }
                (Some(c), None) | (None, Some(c)) => {
                    nodes.push(c.clone());
                }
                (None, None) => (),
            }

            pointer += 1;
        }

        nodes
    }

    fn trace(&self) -> (Vec<Self>, Vec<(usize, usize)>) {
        let mut nodes = vec![self.clone()];
        let mut edges = vec![];
        let mut pointer = 0;

        while nodes.len() > pointer {
            let node = nodes[pointer].0.clone();
            let node = node.lock().unwrap();

            match &node.children {
                (Some(c1), Some(c2)) => {
                    if Arc::ptr_eq(&c1.0, &c2.0) {
                        match nodes
                            .iter()
                            .enumerate()
                            .find(|(_, s)| Arc::ptr_eq(&c1.0, &s.0))
                        {
                            Some((i, _)) => {
                                edges.push((i, pointer));
                                edges.push((i, pointer));
                            }
                            None => {
                                nodes.push(c1.clone());

                                edges.push((nodes.len() - 1, pointer));
                                edges.push((nodes.len() - 1, pointer));
                            }
                        }
                    } else {
                        match nodes
                            .iter()
                            .enumerate()
                            .find(|(_, s)| Arc::ptr_eq(&c1.0, &s.0))
                        {
                            Some((i, _)) => {
                                edges.push((i, pointer));
                            }
                            None => {
                                nodes.push(c1.clone());
                                edges.push((nodes.len() - 1, pointer));
                            }
                        }

                        match nodes
                            .iter()
                            .enumerate()
                            .find(|(_, s)| Arc::ptr_eq(&c2.0, &s.0))
                        {
                            Some((i, _)) => {
                                edges.push((i, pointer));
                            }
                            None => {
                                nodes.push(c2.clone());
                                edges.push((nodes.len() - 1, pointer));
                            }
                        }
                    }
                }
                (Some(c), None) | (None, Some(c)) => {
                    match nodes
                        .iter()
                        .enumerate()
                        .find(|(_, s)| Arc::ptr_eq(&c.0, &s.0))
                    {
                        Some((i, _)) => {
                            edges.push((i, pointer));
                        }
                        None => {
                            nodes.push(c.clone());
                            edges.push((nodes.len() - 1, pointer));
                        }
                    }
                }
                (None, None) => (),
            }

            pointer += 1;
        }

        (nodes, edges)
    }
}

impl<T: Float + NumAssignOps + PartialEq + Display> Scalar<T> {
    pub fn draw(&self) -> String {
        let (nodes, edges) = self.trace();
        let mut vg = VisualGraph::new(Orientation::LeftToRight);

        let node_handles: Vec<(Option<NodeHandle>, NodeHandle)> = nodes
            .iter()
            .map(|node| {
                let node = node.0.lock().unwrap();

                let shape = ShapeKind::new_box(&format!(
                    "{} | data {:.4} | grad {:.4}",
                    node.label, node.data, node.grad
                ));

                let element = Element::create(
                    shape,
                    StyleAttr::simple(),
                    Orientation::LeftToRight,
                    Point::new(250.0, 25.0),
                );

                if let Some(op) = &node.op {
                    let shape = ShapeKind::new_circle(match op {
                        Op::ADD => "+",
                        Op::SUB => "-",
                        Op::MUL => "*",
                        // Op::DIV => "/",
                        Op::POWI(_) => "POWI",
                        Op::TANH => "tanh",
                    });

                    let op_element = Element::create(
                        shape,
                        StyleAttr::simple(),
                        Orientation::LeftToRight,
                        Point::new(40.0, 40.0),
                    );

                    let handle = vg.add_node(element);
                    let op_handle = vg.add_node(op_element);

                    vg.add_edge(Arrow::simple(""), op_handle.clone(), handle);

                    (Some(op_handle), handle)
                } else {
                    (None, vg.add_node(element))
                }
            })
            .collect();

        for (from, to) in edges {
            let (_, from_handle) = node_handles[from];
            let to_handle = match node_handles[to] {
                (Some(handle), _) => handle,
                (None, handle) => handle,
            };
            vg.add_edge(Arrow::simple(""), from_handle, to_handle);
        }

        let mut svg = SVGWriter::new();

        vg.do_it(false, false, false, &mut svg);
        svg.finalize()
    }
}

impl<T: Float + NumAssignOps + PartialEq> PartialEq for Scalar<T> {
    fn eq(&self, other: &Self) -> bool {
        let value = self.0.lock().unwrap();
        let other_value = other.0.lock().unwrap();

        value.data == other_value.data
    }
}

impl<T: Float + NumAssignOps + PartialEq> Eq for Scalar<T> {}

impl<T: Float + NumAssignOps> PartialOrd for Scalar<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let value = self.0.lock().unwrap();
        let other_value = other.0.lock().unwrap();

        value.data.partial_cmp(&other_value.data)
    }
}

impl<T: Float + NumAssignOps + PartialEq + Ord> Ord for Scalar<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let value = self.0.lock().unwrap();
        let other_value = other.0.lock().unwrap();

        value.data.cmp(&other_value.data)
    }
}

impl<T: Add<Output = T> + Float + NumAssignOps> Add for Scalar<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let value = self.0.lock().unwrap();
        let self_data = value.data;
        drop(value);

        let other_value = other.0.lock().unwrap();
        let other_data = other_value.data;
        drop(other_value);

        let mut output = Value::new(self_data + other_data, "");

        output.children = (Some(self.clone()), Some(other.clone()));
        output.op = Some(Op::ADD);

        Scalar(Arc::new(Mutex::new(output)))
    }
}

impl<T: Add<Output = T> + Float + NumAssignOps> AddAssign for Scalar<T> {
    fn add_assign(&mut self, other: Self) {
        let value = self.0.lock().unwrap();
        let self_data = value.data;
        drop(value);

        let other_value = other.0.lock().unwrap();
        let other_data = other_value.data;
        drop(other_value);

        let mut output = Value::new(self_data + other_data, "");

        output.children = (Some(self.clone()), Some(other.clone()));
        output.op = Some(Op::ADD);

        *self = Scalar(Arc::new(Mutex::new(output)));
    }
}

impl<T: Sub<Output = T> + Float + NumAssignOps> Sub for Scalar<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.0.lock().unwrap();
        let self_data = value.data;
        drop(value);

        let rhs_value = rhs.0.lock().unwrap();
        let rhs_data = rhs_value.data;
        drop(rhs_value);

        let mut output = Value::new(self_data - rhs_data, "");

        output.children = (Some(self.clone()), Some(rhs.clone()));
        output.op = Some(Op::SUB);

        Scalar(Arc::new(Mutex::new(output)))
    }
}

impl<T: Mul<Output = T> + Float + NumAssignOps> Mul for Scalar<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let value = self.0.lock().unwrap();
        let self_data = value.data;
        drop(value);

        let rhs_value = rhs.0.lock().unwrap();
        let rhs_data = rhs_value.data;
        drop(rhs_value);

        let mut output = Value::new(self_data * rhs_data, "");

        output.children = (Some(self.clone()), Some(rhs.clone()));
        output.op = Some(Op::MUL);

        Scalar(Arc::new(Mutex::new(output)))
    }
}

// impl<T: Div<Output = T> > Div for Scalar<T> {
//     type Output = Self;

//     fn div(self, rhs: Self) -> Self::Output {
//         let value = self.0.lock().unwrap();
//         let self_data = value.data;
//         drop(value);

//         let rhs_value = rhs.0.lock().unwrap();
//         let rhs_data = rhs_value.data;
//         drop(rhs_value);

//         let mut output = Value::new(self_data / rhs_data, "");

//         output.children = Some((self.clone(), rhs.clone()));
//         output.op = Some(Op::DIV);

//         Scalar(Arc::new(Mutex::new(output)))
//     }
// }

impl<T: Float + NumAssignOps> Scalar<T> {
    pub fn powi(&self, n: i32) -> Self {
        let value = self.0.lock().unwrap();
        let self_data = value.data;
        drop(value);

        let mut output = Value::new(self_data.powi(n), "");

        output.children = (Some(self.clone()), None);
        output.op = Some(Op::POWI(n));

        Scalar(Arc::new(Mutex::new(output)))
    }

    pub fn tanh(&self) -> Self {
        let value = self.0.lock().unwrap();
        let self_data = value.data;
        drop(value);

        let mut output = Value::new(self_data.tanh(), "");

        output.children = (Some(self.clone()), None);
        output.op = Some(Op::TANH);

        Scalar(Arc::new(Mutex::new(output)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Scalar::new(1.0, "a");
        let b = Scalar::new(2.0, "b");

        assert!(a + b == Scalar::new(3.0, ""));

        let a = Scalar::new(1.0, "a");
        assert!(a.clone() + a.clone() == Scalar::new(2.0, ""));
        assert!(a.clone() - a.clone() == Scalar::new(0.0, ""));
        assert!(a.clone() * a.clone() == Scalar::new(1.0, ""));
        // assert!(a.clone() / a.clone() == Scalar::new(1.0, ""));
        // assert!({
        //     let t = a.clone();
        //     t.pow(a.clone()) == Scalar::new(1.0, "")
        // });

        assert!(Scalar::new(1.0, "") - Scalar::new(2.0, "") == Scalar::new(-1.0, ""));
        assert!(Scalar::new(2.0, "") * Scalar::new(3.0, "") == Scalar::new(6.0, ""));
        // assert!(Scalar::new(5.0, "") / Scalar::new(2.0, "") == Scalar::new(2.5, ""));
        // assert!(Scalar::new(2.0, "").pow(Scalar::new(3.0, "")) == Scalar::new(8.0, ""));

        let a = Scalar::new(1.0, "a");
        let b = Scalar::new(2.0, "b");
        let c = Scalar::new(4.0, "c");

        let d = a + b;
        let e = d * c;

        let (nodes, edges) = e.trace();

        assert_eq!(
            nodes
                .iter()
                .map(|n| {
                    let v = n.0.lock().unwrap();
                    v.data
                })
                .collect::<Vec<f32>>(),
            vec![12.0, 3.0, 4.0, 1.0, 2.0]
        );
        assert_eq!(edges, vec![(1, 0), (2, 0), (3, 1), (4, 1)]);
    }
}
