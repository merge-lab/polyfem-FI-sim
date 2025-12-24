#include <filesystem>
#include <iostream>
#include <random>

#include <igl/read_triangle_mesh.h>
#include <igl/readMSH.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_facets.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/remove_unreferenced.h>
#include <igl/Hit.h>
#include <igl/AABB.h>

#include <CLI/CLI.hpp>

#include <nlohmann/json.hpp>

#include <time.h>

using json = nlohmann::json;

bool load_files(const std::string &json_path,
                std::string &model_path,
                std::vector<std::string> &selection_paths,
                std::vector<int> &bc_surfaces,
                std::vector<int> &opt_surfaces,
                std::vector<int> &target_surfaces,
                double &vertex_scaling)
{
    std::ifstream config(json_path);

    if (!config.is_open())
        return false;

    json json_config = json::parse(config);

    model_path = json_config["model_path"].get<std::string>();
    selection_paths = json_config["selection_paths"].get<std::vector<std::string>>();
    bc_surfaces = json_config["bc_surfaces"].get<std::vector<int>>();
    opt_surfaces = json_config["opt_surfaces"].get<std::vector<int>>();
    target_surfaces = json_config["target_surfaces"].get<std::vector<int>>();
    vertex_scaling = json_config["vertex_scaling"].get<double>();

    return true;
}

bool ray_triangle_intersect(
    const Eigen::MatrixXd &V,
    const Eigen::VectorXi &tri,
    const Eigen::Vector3d &origin,
    const Eigen::Vector3d &direction)
{
    // Custom implementation of ray-triangle intersection, for an implementation of 
    // point_inside_mesh and thus face_inside_mesh that doesn't use igl::AABB.

    // The following is from Wikipedia, has eps checks
    const double eps = 1e-8;
    Eigen::Vector3d v0 = V.row(tri(0));
    Eigen::Vector3d v1 = V.row(tri(1));
    Eigen::Vector3d v2 = V.row(tri(2));
    Eigen::Vector3d edge1, edge2, h, s, q;
    double a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = direction.cross(edge2);
    a = edge1.dot(h);
    if (a > -eps && a < eps)
        return false; // This ray is parallel to this triangle.
    f = 1.0 / a;
    s = origin - v0;
    u = f * s.dot(h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = s.cross(edge1);
    v = f * direction.dot(q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    double t = f * edge2.dot(q);
    if (t > eps) // ray intersection
        return true;
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

bool directional_point_inside_mesh(
    igl::AABB<Eigen::MatrixXd, 3> &tree,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &point,
    const Eigen::Vector3d direction)
{
    std::vector<igl::Hit> hits;
    tree.intersect_ray(V, F, point, direction, hits);

    // Count the number of times a ray in the +y axis intersects with the mesh
    // by counting the number of triangles that intersect with the ray
    int triangles_intersected = 0;
    for (const auto &h : hits)
        if ((h.id >= 0) && (h.t > 1e-4))
            triangles_intersected++;

    // If intersect an odd times, point is in mesh
    // If intersect an even times, point is outside mesh
    return triangles_intersected % 2;
}

// Assume mesh is watertight
bool point_inside_mesh(
    igl::AABB<Eigen::MatrixXd, 3> &tree,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &point)
{
    int N_samples = 5;
    std::mt19937 gen;
    gen.seed(1);
    std::normal_distribution<double> dist(0, 1);
    // Create a 3xN matrix, where each column, once normalized, will be our random dir
    Eigen::MatrixXd mat_random_vs = Eigen::MatrixXd::NullaryExpr(3, N_samples, [&](){
        return dist(gen);
    });

    int point_in_mesh_in_dir = 0;
    for (int i = 0; i < mat_random_vs.cols(); i++)
    {
        Eigen::MatrixXd dir_i = mat_random_vs.col(i);
        dir_i.normalize();
        if (directional_point_inside_mesh(tree, V, F, point, dir_i))
            point_in_mesh_in_dir++;
    }
    
    // Return the majority value
    return point_in_mesh_in_dir > (int) N_samples / 2;
}

bool directional_point_inside_mesh(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &point,
    const Eigen::Vector3d direction)
{
    // Count the number of times a ray in the +y axis intersects with the mesh
    // by counting the number of triangles that intersect with the ray
    int triangles_intersected = 0;
    for (int i = 0; i < F.rows(); ++i)
        if (ray_triangle_intersect(V, F.row(i), point, direction))
            triangles_intersected++;

    // If intersect an odd times, point is in mesh
    // If intersect an even times, point is outside mesh
    return triangles_intersected % 2;
}

bool point_inside_mesh(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXd &point)
{
    // Create collection of random directions
    // The best way to uniformly sample unit vectors in R3 (ie sample from the surface of a ball)
    // is to sample Gaussian rv's in R3, and normalize.
    // See https://angms.science/doc/RM/randUnitVec.pdf

    Eigen::Vector3d x_dir, y_dir, z_dir, diag1, diag2, diag3, diag4;
    x_dir << 1, 0, 0;
    y_dir << 0, 1, 0;
    z_dir << 0, 0, 1;
    diag1 << 1, 2, 1;
    diag2 << 1, 0, -2;
    diag3 << -1, -0.1, 1;
    diag4 << -1, 0.1, 1;

    diag1.normalize();
    diag2.normalize();
    diag3.normalize();
    diag4.normalize();

    std::vector<Eigen::Vector3d> directions{x_dir, y_dir, z_dir, diag1, diag2, diag3, diag4};

    // Append a list of (predictable) random vectors
    int N_random = 20;
    std::mt19937 gen;
    gen.seed(1);
    std::normal_distribution<double> dist(0, 1);
    for (int i = 0; i < N_random; i++)
    {
        Eigen::MatrixXd dir_i_rand = Eigen::MatrixXd::NullaryExpr(3, 1, [&](){return dist(gen);});
        dir_i_rand.normalize();
        directions.push_back(dir_i_rand);
    }

    int point_in_mesh_in_dir = 0;
    for (const auto &dir_i : directions) {
        if (directional_point_inside_mesh(V, F, point, dir_i))
            point_in_mesh_in_dir++;
    }

    // Return the majority value
    return point_in_mesh_in_dir > (int) directions.size() / 3;
}

bool face_inside_mesh(
    igl::AABB<Eigen::MatrixXd, 3> &tree,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_face,
    const Eigen::VectorXi &face)
{
    // If all three points of a face are not within a mesh, then the face 
    // is not within the mesh

    // ie check if any part of the face intersects with the mesh at all

    for (int i = 0; i < 3; ++i)
        if (!point_inside_mesh(tree, V, F, V_face.row(face(i))))
            return false;
    return true;
}

bool face_inside_mesh(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &V_face,
    const Eigen::VectorXi &face)
{
    // If all three points of a face are not within a mesh, then the face 
    // is not within the mesh

    // ie check if any part of the face intersects with the mesh at all

    for (int i = 0; i < 3; ++i)
        if (!point_inside_mesh(V, F, V_face.row(face(i))))
            return false;
    return true;
}

void connected_components_in_selection(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const std::string &selection_file,
    std::vector<Eigen::MatrixXi> &F_groups,
    std::set<int> &visited_faces)
{
    Eigen::MatrixXd V_select;
    Eigen::MatrixXi F_select;
    bool read_mesh = igl::read_triangle_mesh(selection_file, V_select, F_select);
    if (!read_mesh)
        throw("could not read selection mesh: " + selection_file);

    std::vector<std::vector<int>> VF, VI;
    igl::vertex_triangle_adjacency(V.rows(), F, VF, VI);
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);

    // Initialize the AABB (Axis-aligned Bounding box) tree, which will enable more efficient
    // ray-mesh intersection calculations
    // Note that this is not actually used - we use the from-scratch implementation at the top instead
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(V_select, F_select);

    // We will visit vertices in the original mesh, until we have visited all of them
    // To do so, we will create a list of all mesh vertices, and remove items that we visit,
    // until there are none left.

    // Initialize visit-tracking list
    std::set<int> unvisited_vertices;
    for (int i = 0; i < V.rows(); ++i)
        unvisited_vertices.insert(i);

    // While there are still more vertices to visit...
    while (!unvisited_vertices.empty())
    {
        // Mark off the new vertex from the list of unvisited vertices
        int idx = *unvisited_vertices.begin();
        unvisited_vertices.erase(unvisited_vertices.begin());

        if (!point_inside_mesh(V_select, F_select, V.row(idx)))
            // If point is not inside selection mesh, do nothing and continue to next vertex
            continue;

        Eigen::MatrixXi F_;
        std::vector<int> stack;
        for (const auto &f : VF[idx])
            // For each face that is adjacent to the current node ...?
            // Check whether that face lies within the selection mesh
            if (face_inside_mesh(V_select, F_select, V, F.row(f)) && (visited_faces.count(f) == 0))
            {
                // If that face does intersect with / lie within the selection mesh, stache it
                stack.push_back(f);
                break;
            }
        if (stack.empty())
        {
            // std::cout << "Found lone vertex in selection!" << std::endl;

            // No faces adjacent to this vertex intersect with the selection mesh, so do
            //  nothing and continue to next vertex
            continue;
        }
        
        // Check that stack size should never be empty (why? (because if it isn't we'd have continued))
        assert(stack.size() == 1);

        while (!stack.empty())
        {
            // Grab from the stack a triangle face that intersects with the selection mesh
            int f = stack.back();
            stack.pop_back();
            for (int i = 0; i < 3; ++i)
                // For each triangle adjacent to the current face:
                if (visited_faces.count(TT(f, i)) == 0) // If TT(f, i) has not been visited (what happens if it has?)
                    // Todo - understand this
                    if (face_inside_mesh(V_select, F_select, V, F.row(TT(f, i))))
                        stack.push_back(TT(f, i));

            visited_faces.insert(f); // Make sure to (re)-visit f
            F_.conservativeResize(F_.rows() + 1, 3);
            F_.row(F_.rows() - 1) = F.row(f);
        }

        for (int i = 0; i < F_.rows(); ++i)
            for (int j = 0; j < F_.cols(); ++j)
                unvisited_vertices.erase(F_(i, j));

        F_groups.push_back(F_);
    }
}

void write_surface_selection(std::ofstream &file, const Eigen::MatrixXi &F, const int idx)
{
    for (int i = 0; i < F.rows(); ++i)
        file << std::to_string(idx) + " " + std::to_string(F(i, 0)) + " " + std::to_string(F(i, 1)) + " " + std::to_string(F(i, 2)) << std::endl;
}

void write_surface_mesh(const int i, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_, I_;
    igl::remove_unreferenced(V, F, V_, F_, I_);

    std::cout << "writing mesh " << i << std::endl;
    // igl::writeOBJ("mesh" + std::to_string(i) + ".obj", V_, F_);
    std::cout << "Size of V " << V_.rows() << " " << V_.cols() << std::endl;
    std::cout << "Size of F " << F_.rows() << " " << F_.cols() << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    // std::string model_path = "../cantilever_test_beam_cutout.msh";
    // std::vector<std::string> selection_paths = {
    //     "../cantilever_test_external_load.STL",
    //     "../cantilever_test_fixed_bc.STL",
    //     "../cantilever_test_movable_vertices.STL"};
    // std::vector<int> bc_surfaces = {0, 1};
    // std::vector<int> opt_surfaces = {2};
    // std::vector<int> target_surfaces = {};

    // std::string model_path = "../gripper_03_hirez-clean.msh";
    // std::vector<std::string> selection_paths = {
    //     "../gripper_03_fixed_bc.STL",
    //     "../gripper_03_outer_top.STL",
    //     "../gripper_03_inner_top.STL",
    //     "../gripper_03_inner_bottom.STL",
    //     "../gripper_03_fingerpad.STL"};
    // std::vector<int> bc_surfaces = {0, 3, 2};
    // std::vector<int> opt_surfaces = {1};
    // std::vector<int> target_surfaces = {4};

    // std::string model_path = "../makesense_gripper_main.msh";
    // std::vector<std::string> selection_paths = {
    //     "../makesense_gripper_fixed_bc.STL",
    //     "../makesense_gripper_pressure_bc.STL"};
    // std::vector<int> bc_surfaces = {0, 1};
    // std::vector<int> opt_surfaces = {};
    // std::vector<int> target_surfaces = {};

    // std::string model_path = "../../frog/frog_base_mesh.msh";
    // std::vector<std::string> selection_paths = {
    //     "../../frog/frog_fixed_bc.STL",
    //     "../../frog/frog_pressure_bc.STL"};
    // std::vector<int> bc_surfaces = {0, 1};
    // std::vector<int> opt_surfaces = {};
    // std::vector<int> target_surfaces = {};

    // std::string model_path = "../../finger/finger_base_mesh.msh";
    // std::vector<std::string> selection_paths = {
    //     "../../finger/finger_fixed_bc.obj",
    //     "../../finger/finger_pressure_bc.obj",
    //     "../../finger/finger_target_selection.obj"};
    // std::vector<int> bc_surfaces = {0, 1};
    // std::vector<int> opt_surfaces = {};
    // std::vector<int> target_surfaces = {2};

    CLI::App command_line{"setup_3d"};

    command_line.ignore_case();
    command_line.ignore_underscore();

    std::string json_path = "";
    command_line.add_option("--json", json_path, "Path to the json configuration file.");

    CLI11_PARSE(command_line, argc, argv);

    std::string model_path;
    std::vector<std::string> selection_paths;
    std::vector<int> bc_surfaces, opt_surfaces, target_surfaces;
    double vertex_scaling;

    if (!load_files(json_path, model_path, selection_paths, bc_surfaces, opt_surfaces, target_surfaces, vertex_scaling))
        throw std::invalid_argument("Json file not found!");

    Eigen::MatrixXd V;
    Eigen::MatrixXi F, T;
    Eigen::VectorXi FTag, TTag;
    bool read_mesh = igl::readMSH(model_path, V, F, T, FTag, TTag);
    if (!read_mesh)
        throw("could not read mesh: " + model_path);

    V *= vertex_scaling;

    igl::boundary_facets(T, F);
    // std::cout << "Size of T " << T.rows() << " " << T.cols() << std::endl;
    // std::cout << "Size of F " << F.rows() << " " << F.cols() << std::endl;

    std::vector<Eigen::MatrixXi> F_groups;
    // connected_components_in_selection(V, F, selection_paths[0], F_groups);

    std::set<int> visited_faces = {};

    std::ofstream file("surface_selections.txt");
    if (!file.is_open())
        throw "Cannot open file for writing grouped edge list.";

    // for (int i = 0; i < grouped_edge_list.size(); ++i)
    //     for (auto edge : grouped_edge_list[i])
    //         file << i + 1 << " " << edge.first << " " << edge.second << std::endl;
    int index = 1;
    for (const auto &s : bc_surfaces)
    {
        F_groups = {};
        connected_components_in_selection(V, F, selection_paths[s], F_groups, visited_faces);
        for (const auto &group : F_groups)
        {
            std::cout << "BC surface has boundary id " << index << std::endl;
            write_surface_selection(file, group, index);

            write_surface_mesh(index, V, group);

            index++;
        }
    }

    for (const auto &s : opt_surfaces)
    {
        F_groups = {};
        connected_components_in_selection(V, F, selection_paths[s], F_groups, visited_faces);
        for (const auto &group : F_groups)
        {
            std::cout << "OPT surface has boundary id " << index << std::endl;
            write_surface_selection(file, group, index);

            write_surface_mesh(index, V, group);

            index++;
        }
    }

    for (const auto &s : target_surfaces)
    {
        F_groups = {};
        connected_components_in_selection(V, F, selection_paths[s], F_groups, visited_faces);
        for (const auto &group : F_groups)
        {
            std::cout << "TARGET surface has boundary id " << index << std::endl;
            write_surface_selection(file, group, index);

            write_surface_mesh(index, V, group);

            index++;
        }
    }

    return 0;
}