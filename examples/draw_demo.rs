//! A good way of displaying an SVG image in egui.
//!
//! Requires the dependency `egui_extras` with the `svg` feature.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use layout::core::utils::save_to_file;
use micrograd::{engine::Scalar, nn::MLP};
use resvg::{
    tiny_skia,
    usvg::{FitTo, Options, Tree},
    usvg_text_layout::{fontdb, TreeTextToPath},
};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1500.0, 700.0)),
        ..Default::default()
    };
    eframe::run_native(
        "micrograd-rs draw demo",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}

struct MyApp {
    svg_image: egui_extras::RetainedImage,
}

impl Default for MyApp {
    fn default() -> Self {
        // let x1 = Scalar::new(2.0, "x1");
        // let x2 = Scalar::new(0.0, "x2");
        // let w1 = Scalar::new(-3.0, "w1");
        // let w2 = Scalar::new(1.0, "w2");

        // let b = Scalar::new(6.8813735870195432, "b");

        // let x1w1 = x1 * w1;
        // x1w1.label("x1w1");
        // let x2w2 = x2 * w2;
        // x2w2.label("x2w2");

        // let x1w1x2w2 = x1w1 + x2w2;
        // x1w1x2w2.label("x1*w1 + x2*w2");

        // let n = x1w1x2w2 + b;
        // n.label("n");

        // let o = n.tanh();
        // o.label("o");

        // o.backward();

        // let a = Scalar::new(3.0, "a");
        // let b = a.clone() + a;
        // b.label("b");
        // b.backward();

        let a = Scalar::new(-2.0, "a");
        let b = Scalar::new(3.0, "b");
        let d = a.clone() * b.clone();
        d.label("d");
        let e = a + b;
        e.label("e");
        let f = d * e;
        f.label("f");
        f.backward();

        // let x = vec![2.0, 3.0, -1.0]
        //     .iter()
        //     .map(|n| Scalar::new(*n, ""))
        //     .collect();
        // let mut n = MLP::new(3, &[4, 4, 1]);
        // let o = n.output(x).unwrap();

        // save_to_file("/tmp/demo.svg", &o[0].draw()).unwrap();
        // println!("Save demo svg image in /tmp/demo.svg");

        let mut tree = Tree::from_str(&f.draw(), &Options::default()).unwrap();
        let mut fontdb = fontdb::Database::new();

        fontdb.load_system_fonts();
        tree.convert_text(&fontdb);

        let fit_to = FitTo::Zoom(1.0);
        let pixmap_size = fit_to.fit_to(tree.size.to_screen_size()).unwrap();
        let mut pixmap = tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height()).unwrap();

        resvg::render(
            &tree,
            fit_to,
            tiny_skia::Transform::default(),
            pixmap.as_mut(),
        )
        .unwrap();

        Self {
            svg_image: egui_extras::RetainedImage::from_image_bytes(
                "nodes_image",
                &pixmap.encode_png().unwrap(),
            )
            .unwrap(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Nodes draw demo");

            ui.separator();

            let max_size = ui.available_size();
            self.svg_image.show_size(ui, max_size);
        });
    }
}
