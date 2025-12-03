import './App.css'
import Dashboard from "./Dashboard.tsx";
import {QueryClient, QueryClientProvider} from "@tanstack/react-query";

const client = new QueryClient();

function App() {

    return (
        <QueryClientProvider client={client}>
            <Dashboard />
        </QueryClientProvider>
    )
}

export default App